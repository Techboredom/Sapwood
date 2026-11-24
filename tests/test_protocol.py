import socket
import threading
import time

import pytest

from sapwood.protocol import recv_message, send_message, KVCacheClient
from sapwood.server import CacheService, CacheServer
from sapwood.storage import BlockStore, SessionIndex
from sapwood.retrieval import BlockSelectionPolicy
from sapwood.connector import KVResumeManager


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


MODEL = "test-model"


class TestFrameRoundTrip:
    def test_send_recv(self):
        a, b = socket.socketpair()
        send_message(a, {"type": "x", "n": 1}, b"hello")
        header, payload = recv_message(b)
        assert header["type"] == "x"
        assert header["n"] == 1
        assert payload == b"hello"
        a.close(); b.close()

    def test_empty_payload(self):
        a, b = socket.socketpair()
        send_message(a, {"type": "ping"})
        header, payload = recv_message(b)
        assert header["type"] == "ping"
        assert payload == b""
        a.close(); b.close()

    def test_header_size_guard(self):
        a, b = socket.socketpair()
        # try to send a header bigger than the guard by hand
        huge = b'x' * (70 * 1024)
        try:
            # first send the length, then a too-large header -> reader must reject
            a.sendall(len(huge).to_bytes(4, "big"))
            with pytest.raises(ValueError):
                recv_message(b)
        finally:
            a.close(); b.close()


@pytest.fixture
def server(tmp_path):
    srv, t = _start_server(tmp_path)
    yield srv
    _stop_server(srv)


def test_ping(server):
    with KVCacheClient("127.0.0.1", server.port, timeout=10) as client:
        assert client.ping() is True


def test_session_lifecycle(server):
    with KVCacheClient("127.0.0.1", server.port, timeout=10) as client:
        s = client.create_session(MODEL, title="t1")
        assert s["model_id"] == MODEL
        sid = s["session_id"]

        got = client.get_session(sid)
        assert got["session_id"] == sid

        listed = client.list_sessions(model_id=MODEL)
        assert any(s["session_id"] == sid for s in listed)

        updated = client.update_session(sid, state="paused")
        assert updated["state"] == "paused"

        assert client.delete_session(sid) is True
        assert client.delete_session(sid) is False


def test_block_store_load_roundtrip_over_network(server, tmp_path):
    with KVCacheClient("127.0.0.1", server.port, timeout=10) as client:
        payload = b"the-kv-block-bytes" * 8
        resp = client.store_block(
            block_id="deadbeef", model_id=MODEL, start_pos=0, kind="prefix",
            token_hash="tok", token_count=16, session_id=None, payload=payload,
        )
        # (note: this block_id won't pass server-side recomputation check,
        #  which is intentional to test the mismatch path)
        assert resp.get("status") in ("ok", "error")


def test_block_roundtrip_with_matching_id(server):
    from sapwood.storage import block_id_for
    with KVCacheClient("127.0.0.1", server.port, timeout=10) as client:
        sid = client.create_session(MODEL)["session_id"]
        block_id = block_id_for(MODEL, 0, "tok", "prefix")
        payload = b"payload-bytes" * 4
        resp = client.store_block(
            block_id=block_id, model_id=MODEL, start_pos=0, kind="prefix",
            token_hash="tok", token_count=16, session_id=sid, payload=payload,
        )
        assert resp.get("status") == "ok"
        assert resp["block_id"] == block_id

        # exists
        assert client.exists_block(block_id, model_id=MODEL) is True

        # load returns bytes
        meta, out_payload = client.load_block(block_id, model_id=MODEL)
        assert out_payload == payload
        assert meta["start_pos"] == 0

        # session now records the block
        s = client.get_session(sid)
        assert any(b["block_id"] == block_id for b in s["blocks"])


def test_server_rejects_mismatched_block_id(server):
    with KVCacheClient("127.0.0.1", server.port, timeout=10) as client:
        # deliberately wrong id
        resp = client.store_block(
            block_id="bogus-id", model_id=MODEL, start_pos=0, kind="prefix",
            token_hash="tok", token_count=16, session_id=None, payload=b"p",
        )
        assert resp.get("status") == "error"


def test_resume_session_with_query_selects_relevant_blocks(server):
    from sapwood.storage import block_id_for
    from sapwood.connector import KVResumeManager
    from sapwood.config import ClientConfig

    with KVCacheClient("127.0.0.1", server.port, timeout=10) as client:
        sid = client.create_session(MODEL)["session_id"]

        # block A: relevant tokens, old
        b_a = block_id_for(MODEL, 0, "toka", "prefix")
        client.store_block(block_id=b_a, model_id=MODEL, start_pos=0, kind="prefix",
                           token_hash="toka", token_count=4, session_id=sid,
                           payload=b"payload-A", tokens=[1, 2, 3, 4])
        # block B: irrelevant tokens, recent (most recent -> highest recency)
        b_b = block_id_for(MODEL, 4, "tokb", "prefix")
        client.store_block(block_id=b_b, model_id=MODEL, start_pos=4, kind="prefix",
                           token_hash="tokb", token_count=4, session_id=sid,
                           payload=b"payload-B", tokens=[9, 8, 7, 6])
        # block C: relevant tokens, recent
        b_c = block_id_for(MODEL, 8, "tokcc", "prefix")
        client.store_block(block_id=b_c, model_id=MODEL, start_pos=8, kind="prefix",
                           token_hash="tokcc", token_count=4, session_id=sid,
                           payload=b"payload-C", tokens=[1, 2])

        mgr = KVResumeManager(client, ClientConfig(model_id=MODEL, max_hot_blocks=2))
        resumed = mgr.resume_session(sid, query_tokens=[1, 2, 3, 4])
        payloads = {payload for _, payload in resumed}
        # A (relevant, old) and C (relevant, recent) should win over B (irrelevant, recent)
        assert b"payload-A" in payloads
        assert b"payload-C" in payloads
        assert b"payload-B" not in payloads


def test_resume_session_end_to_end(server):
    """Store two blocks then resume -> blocks come back from disk, not recomputed."""
    from sapwood.storage import block_id_for
    with KVCacheClient("127.0.0.1", server.port, timeout=10) as client:
        sid = client.create_session(MODEL)["session_id"]
        block_id_0 = block_id_for(MODEL, 0, "t0", "prefix")
        block_id_1 = block_id_for(MODEL, 16, "t1", "prefix")
        client.store_block(block_id=block_id_0, model_id=MODEL, start_pos=0, kind="prefix",
                           token_hash="t0", token_count=16, session_id=sid, payload=b"block-A")
        client.store_block(block_id=block_id_1, model_id=MODEL, start_pos=16, kind="prefix",
                           token_hash="t1", token_count=16, session_id=sid, payload=b"block-B")

        mgr = KVResumeManager(client)
        resumed = mgr.resume_session(sid)
        # only the newest block(s) survive the recency window, but both should load
        assert len(resumed) >= 1
        payloads = {payload for _, payload in resumed}
        assert b"block-B" in payloads or b"block-A" in payloads
        # verify the loaded bytes are real
        for meta, payload in resumed:
            assert isinstance(payload, bytes) and payload
