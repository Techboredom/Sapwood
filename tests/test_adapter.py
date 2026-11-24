"""Tests for the vLLM connector adapter (sapwood.vllm_adapter).

The connector is UNVERIFIED against a real vLLM engine — these tests
exercise the scheduler/worker logic against a real in-process sapwood server,
with mocked vLLM objects (vllm_config, kv_cache_config, blocks) so no GPU or
vLLM install is required.
"""

import threading
import types

import pytest

from sapwood.server import CacheService, CacheServer
from sapwood.storage import BlockStore, SessionIndex, block_id_for
from sapwood.vllm_adapter import (
    SapwoodScheduler,
    SapwoodWorker,
    SapwoodConnectorMetadata,
    _bundle_pages,
    _unbundle_pages,
    _page_bytes,
    _write_page,
    _ordinal_in_group,
    _group_blocks,
    _all_groups_blocks,
)

MODEL = "test-model"


def _start_server(tmp_path):
    store = BlockStore(tmp_path)
    sessions = SessionIndex(tmp_path)
    svc = CacheService(store, sessions)
    server = CacheServer(svc, host="127.0.0.1", port=0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, t


def _stop_server(server):
    server.stop()


@pytest.fixture
def server(tmp_path):
    srv, _ = _start_server(tmp_path)
    yield srv
    _stop_server(srv)


# -- mock objects ------------------------------------------------------------

class FakeBlock:
    """Stand-in for vLLM's KVCacheBlock."""

    def __init__(self, block_id, block_hash=None, is_null=False):
        self.block_id = block_id
        self.block_hash = block_hash
        self.is_null = is_null


class FakeKVCacheConfig:
    """Stand-in for vLLM's KVCacheConfig (single group, uniform block_size)."""

    def __init__(self, block_size=16, layer_names=None):
        self.kv_cache_groups = [
            types.SimpleNamespace(
                kv_cache_spec=types.SimpleNamespace(block_size=block_size),
                layer_names=layer_names if layer_names is not None else ["layer.0"],
            )
        ]


class FakeBlocks:
    """Stand-in for vLLM's KVCacheBlocks (one group)."""

    def __init__(self, blocks):
        self.blocks = (list(blocks),) if blocks else ()


class FakeConfig:
    """Stand-in for vLLM's vllm_config."""

    def __init__(self, host="127.0.0.1", port=7676, model_id=MODEL):
        self.kv_transfer_config = types.SimpleNamespace(
            kv_connector_extra_config={
                "server_host": host,
                "server_port": port,
                "timeout": 60.0,
                "model_id": model_id,
            }
        )


def _fake_request(req_id="req-1", num_computed_tokens=0):
    r = types.SimpleNamespace()
    r.request_id = req_id
    r.num_computed_tokens = num_computed_tokens
    r.num_tokens = num_computed_tokens + 16
    r.num_prompt_tokens = 16
    r.kv_transfer_params = {}
    r.resumable = False
    r.skip_reading_external_kv = False
    return r


# -- scheduler -------------------------------------------------------------

def test_scheduler_builds_store_plan(tmp_path):
    server, t = _start_server(tmp_path)
    try:
        server_config = {"host": "127.0.0.1", "port": server.port, "timeout": 10.0, "model_id": MODEL, "kind": "kv"}
        sched = SapwoodScheduler(FakeConfig(server.port), FakeKVCacheConfig(block_size=16), server_config)
        req = _fake_request()
        blocks = FakeBlocks([FakeBlock(1, "hash-0"), FakeBlock(2, "hash-1")])
        sched._build_store_plans(req, blocks)
        plans = sched._pending_stores[req.request_id]
        assert len(plans) == 2
        assert plans[0].logical_block_id == block_id_for(MODEL, 0, "hash-0", "kv")
        assert plans[1].logical_block_id == block_id_for(MODEL, 16, "hash-1", "kv")
        assert plans[0].gpu_block_ids == {0: 1}
        assert plans[0].token_count == 16
    finally:
        _stop_server(server)


def test_scheduler_skips_blocks_without_hash_or_id(tmp_path):
    server, t = _start_server(tmp_path)
    try:
        server_config = {"host": "127.0.0.1", "port": server.port, "timeout": 10.0, "model_id": MODEL, "kind": "kv"}
        sched = SapwoodScheduler(FakeConfig(server.port), FakeKVCacheConfig(), server_config)
        req = _fake_request()
        blocks = FakeBlocks([FakeBlock(1, None), FakeBlock(0, "hash-1"), FakeBlock(3, "hash-2", is_null=True)])
        sched._build_store_plans(req, blocks)
        # hashless block, null block, and zero block_id are all skipped
        assert sched._pending_stores[req.request_id] == []
    finally:
        _stop_server(server)


def test_scheduler_store_is_idempotent(tmp_path):
    server, t = _start_server(tmp_path)
    try:
        server_config = {"host": "127.0.0.1", "port": server.port, "timeout": 10.0, "model_id": MODEL, "kind": "kv"}
        sched = SapwoodScheduler(FakeConfig(server.port), FakeKVCacheConfig(), server_config)
        req = _fake_request()
        blocks = FakeBlocks([FakeBlock(1, "hash-0")])
        sched._build_store_plans(req, blocks)
        sched._build_store_plans(req, blocks)  # same blocks again
        plans = sched._pending_stores[req.request_id]
        assert len(plans) == 1  # deduped by logical block id
    finally:
        _stop_server(server)


def test_get_num_new_matched_tokens_counts_contiguous_prefix(tmp_path):
    server, t = _start_server(tmp_path)
    try:
        server_config = {"host": "127.0.0.1", "port": server.port, "timeout": 10.0, "model_id": MODEL, "kind": "kv"}
        sched = SapwoodScheduler(FakeConfig(server.port), FakeKVCacheConfig(), server_config)
        req = _fake_request()
        sid = sched._session_for(req)
        # seed the server session with 3 stored blocks
        sched._client.create_session(model_id=MODEL, session_id=sid)
        for i in range(3):
            sched._client.store_block(
                block_id=block_id_for(MODEL, i * 16, f"hash-{i}", "kv"),
                model_id=MODEL, start_pos=i * 16, kind="kv",
                token_hash=f"hash-{i}", token_count=16,
                session_id=sid, payload=b"x",
            )
            # hmm: store_block with session_id adds the block to the session
        # count the prefix beyond 0 computed tokens
        num, async_flag = sched.get_num_new_matched_tokens(req, 0)
        assert num == 48
        assert async_flag is False
    finally:
        _stop_server(server)


def test_get_num_new_matched_tokens_gap_breaks_count(tmp_path):
    server, t = _start_server(tmp_path)
    try:
        server_config = {"host": "127.0.0.1", "port": server.port, "timeout": 10.0, "model_id": MODEL, "kind": "kv"}
        sched = SapwoodScheduler(FakeConfig(server.port), FakeKVCacheConfig(), server_config)
        req = _fake_request()
        sid = sched._session_for(req)
        sched._client.create_session(model_id=MODEL, session_id=sid)
        sched._client.store_block(
            block_id=block_id_for(MODEL, 0, "hash-0", "kv"),
            model_id=MODEL, start_pos=0, kind="kv",
            token_hash="hash-0", token_count=16,
            session_id=sid, payload=b"x",
        )
        sched._client.store_block(
            block_id=block_id_for(MODEL, 32, "hash-2", "kv"),
            model_id=MODEL, start_pos=32, kind="kv",
            token_hash="hash-2", token_count=16,
            session_id=sid, payload=b"x",
        )
        # gap at start_pos=16: only the first block is contiguous
        num, _ = sched.get_num_new_matched_tokens(req, 0)
        assert num == 16
    finally:
        _stop_server(server)


def test_build_connector_meta_clears_pending(tmp_path):
    server, t = _start_server(tmp_path)
    try:
        server_config = {"host": "127.0.0.1", "port": server.port, "timeout": 10.0, "model_id": MODEL, "kind": "kv"}
        sched = SapwoodScheduler(FakeConfig(server.port), FakeKVCacheConfig(), server_config)
        req = _fake_request()
        blocks = FakeBlocks([FakeBlock(1, "hash-0")])
        sched._build_store_plans(req, blocks)
        meta = sched.build_connector_meta(types.SimpleNamespace(finished_req_ids=None))
        assert isinstance(meta, SapwoodConnectorMetadata)
        assert len(meta.store_plan) == 1
        # the pending plans are consumed by build_connector_meta
        assert sched._pending_stores == {}
        meta2 = sched.build_connector_meta(types.SimpleNamespace(finished_req_ids=None))
        assert meta2.store_plan == []
    finally:
        _stop_server(server)


# -- worker: bundle round-trip ---------------------------------------------

def test_bundle_pages_round_trip():
    pages = {"layer.0": b"\x01\x02\x03", "layer.1": b"\x04\x05" * 100}
    out = _bundle_pages(pages)
    assert _unbundle_pages(out) == pages


def test_unbundle_rejects_trailing_garbage():
    pages = {"layer.0": b"\x01"}
    out = _bundle_pages(pages) + b"junk"
    assert _unbundle_pages(out) == {}


def test_page_bytes_and_write_page_round_trip():
    torch = pytest.importorskip("torch")

    kv = torch.arange(2 * 4 * 8, dtype=torch.bfloat16).reshape(2, 4, 8)  # 2 blocks
    page = _page_bytes(kv, 1)
    assert page is not None
    kv2 = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    _write_page(kv2, 1, page)
    assert torch.equal(kv2[1], kv[1])


def test_page_bytes_out_of_range_returns_none():
    torch = pytest.importorskip("torch")

    kv = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    assert _page_bytes(kv, 99) is None


# -- worker: full store/load round trip through a real server ---------------

class FakeKvLayer:
    """A minimal stand-in for a paged KV tensor."""

    def __init__(self, num_blocks, numel):
        torch = pytest.importorskip("torch")

        self.t = torch.zeros(num_blocks, numel, dtype=torch.bfloat16)

    def view(self, *shape):
        return self.t.view(*shape)

    def copy_(self, other):
        self.t.copy_(other)


def test_worker_store_then_load_round_trip(tmp_path):
    server, t = _start_server(tmp_path)
    try:
        server_config = {"host": "127.0.0.1", "port": server.port, "timeout": 10.0, "model_id": MODEL, "kind": "kv"}
        # worker registers the paged cache
        worker = SapwoodWorker(FakeConfig(server.port), FakeKVCacheConfig(), server_config)
        kv_layer = FakeKvLayer(4, 8)
        worker.kv_caches = {"layer.0": kv_layer}
        worker._layer_to_group = {"layer.0": 0}
        # build a store plan (one block, one group)
        logical = block_id_for(MODEL, 0, "hash-0", "kv")
        meta = SapwoodConnectorMetadata(
            store_plan=[
                types.SimpleNamespace(  # type: ignore[attr-defined]
                    req_id="req-1",
                    logical_block_id=logical,
                    start_pos=0,
                    token_hash="hash-0",
                    token_count=16,
                    kind="kv",
                    gpu_block_ids={0: 2},
                )
            ],
            load_plan=[],
        )
        worker.set_metadata(meta)
        # populate the page with known bytes
        torch = pytest.importorskip("torch")

        for i in range(8):
            kv_layer.t[2, i] = torch.tensor(i + 1.0, dtype=torch.bfloat16)
        worker.save_kv_layer("layer.0", kv_layer, None)
        worker.wait_for_save()
        # the server should now hold the block
        from sapwood.protocol import KVCacheClient

        with KVCacheClient("127.0.0.1", server.port, timeout=10) as client:
            assert client.exists_block(logical, MODEL)
        # now load it back and check
        meta2 = SapwoodConnectorMetadata(
            store_plan=[],
            load_plan=[
                types.SimpleNamespace(
                    req_id="req-1",
                    logical_block_id=logical,
                    start_pos=0,
                    token_count=16,
                    gpu_block_ids={0: 3},
                )
            ],
        )
        kv_layer2 = FakeKvLayer(4, 8)
        worker.kv_caches = {"layer.0": kv_layer2}
        worker.set_metadata(meta2)
        worker.start_load_kv(None)
        # block 3 of kv_layer2 should now hold the stored bytes
        torch = pytest.importorskip("torch")

        for i in range(8):
            assert kv_layer2.t[3, i].item() == pytest.approx(i + 1.0, abs=0.5)
    finally:
        _stop_server(server)


# -- group/block helpers -----------------------------------------------------

def test_group_blocks_and_ordinal_in_group():
    blocks = FakeBlocks([FakeBlock(10, "h0"), FakeBlock(11, "h1")])
    group = _group_blocks(blocks, 0)
    assert len(group) == 2
    assert _ordinal_in_group(blocks, 0, group[1]) == 1
    assert _ordinal_in_group(blocks, 0, FakeBlock(12)) is None
    flat = _all_groups_blocks(blocks)
    assert len(flat) == 2
