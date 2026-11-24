"""vLLM plugin integration (KV connector) — real API, UNVERIFIED.

vLLM 0.34.0rc1 (``vllm/vllm-openai:nightly-f8b29fab414884206460fe63b983ca34fc6e3e9b``).

This module subclasses the real ``KVConnectorBase_V1`` and wires the sapwood
disk cache in as the external KV tier for a vLLM worker. It is deliberately a
narrow bridge:

* **Store** (worker): as blocks are computed, the per-layer page bytes for
  the blocks in the current store plan are shipped to the sapwood server.
  Each logical block is a self-contained payload (one page per registered
  layer), so a single block carries everything needed to resume it.
* **Load** (worker): at the start of a forward pass, the blocks in the
  current load plan are fetched from the server and written into the paged
  KV buffer. The scheduler drives this through ``get_num_new_matched_tokens``.

What this does NOT do (and will not claim to do):

* The retrieval-assembly / RAG-window selection that governs *which* blocks
  are GPU-resident is a client-side concept (``connector.py``). This
  connector moves bytes; it does not make hot-window decisions.
* The CacheBlend-style selective-recompute decision is likewise client-side
  (``cacheblend.py``). Only block bytes move through this connector.
* It does not replicate vLLM's full offloading connector (sliding-window /
  recurrent-state draining, EAGLE/MTP groups, partial tails, KV-group
  alignment, GPU-block fences, async stores). It is single-step, full-prefix,
  synchronous, best-effort.
* The short-conv state of Inkling-class models is stored as an ordinary
  KV-cache layer (it is registered as a ``SlidingWindowSpec`` in vLLM), so
  nothing about it is special-cased here.

Unavailable / unverified honest status
--------------------------------------
This file has never been executed against a running vLLM engine (the container
is not in this environment). The signatures are taken from the verified
``vllm/distributed/kv_transfer/kv_connector/v1/base.py`` in the 0.34.0rc1
source tree, but the whole integration end-to-end is **NOT VERIFIED** and is
not a production feature. Do not claim it works until it has been run against
the actual model in the target container.

Registration
------------
Register via ``--kv-transfer-config``::

    vllm serve <model> --kv-transfer-config '{
        "kv_connector": "SapwoodConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "sapwood.vllm_adapter:SapwoodKVConnector",
        "kv_connector_extra_config": {
            "server_host": "127.0.0.1",
            "server_port": 7676,
            "timeout": 60.0,
            "model_id": "thinkingmachines/Inkling-2-Mini"
        }
    }'

The ``sapwood-kv-server`` must be running and reachable at the configured
``server_host``/``server_port``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .protocol import KVCacheClient
from .storage import block_id_for

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorMetadata as _BaseMetadata,
    )

# ---------------------------------------------------------------------------
# Lazy vLLM import — the package must import without vLLM installed.
# ---------------------------------------------------------------------------

_VLLM_AVAILABLE = False
try:  # pragma: no cover - exercised only under vLLM
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
        KVConnectorWorkerMetadata,  # kept for API parity; unused here
    )

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - vLLM absent
    KVConnectorBase_V1 = object  # type: ignore[assignment,misc]
    KVConnectorMetadata = object  # type: ignore[assignment,misc]
    KVConnectorRole = None  # type: ignore[assignment]
    KVConnectorWorkerMetadata = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Configuration (read from kv_connector_extra_config)
# ---------------------------------------------------------------------------


def _extra_config(vllm_config) -> dict[str, Any]:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return {}
    try:
        value = kv_transfer_config.kv_connector_extra_config
        return dict(value or {})
    except AttributeError:  # pragma: no cover
        return {}


def _read_server_config(vllm_config) -> dict[str, Any]:
    extra = _extra_config(vllm_config)
    return {
        "host": str(extra.get("server_host", "127.0.0.1")),
        "port": int(extra.get("server_port", 7676)),
        "timeout": float(extra.get("timeout", 60.0)),
        "model_id": str(extra.get("model_id", "unknown")),
        "kind": str(extra.get("kind", "kv")),
    }


# ---------------------------------------------------------------------------
# Metadata shared scheduler -> worker (in-process, not serialized)
# ---------------------------------------------------------------------------


@dataclass
class SapwoodStorePlan:
    """One logical block the worker should ship to the sapwood server."""

    req_id: str
    logical_block_id: str
    start_pos: int
    token_hash: str
    token_count: int
    kind: str
    # group_idx -> gpu_block_id allocated for this block in that group
    gpu_block_ids: dict[int, int] = field(default_factory=dict)


@dataclass
class SapwoodLoadPlan:
    """One logical block the worker should fetch from the server and write
    into the paged buffer."""

    req_id: str
    logical_block_id: str
    start_pos: int
    token_count: int
    # group_idx -> gpu_block_id allocated for this block in that group
    gpu_block_ids: dict[int, int] = field(default_factory=dict)


@dataclass
class SapwoodConnectorMetadata(KVConnectorMetadata):  # type: ignore[misc]
    """Scheduler -> Worker metadata carrying the load and store plans."""

    store_plan: list[SapwoodStorePlan] = field(default_factory=list)
    load_plan: list[SapwoodLoadPlan] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers: per-layer page extraction / serialization
# ---------------------------------------------------------------------------


def _page_bytes(kv_layer, block_id: int) -> bytes | None:
    """Flatten one block of a paged KV tensor into raw bytes."""
    try:
        import torch

        num_blocks = int(kv_layer.shape[0])
        if block_id < 0 or block_id >= num_blocks:
            return None
        row = kv_layer.view(num_blocks, -1)[block_id]
        # bfloat16 tensors cannot go through numpy directly; reinterpret as
        # uint8 first, then serialize.
        return (
            row.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    except Exception:
        return None


def _write_page(kv_layer, block_id: int, page: bytes) -> None:
    """Write raw page bytes into a paged KV tensor row."""
    try:
        import torch

        num_blocks = int(kv_layer.shape[0])
        if block_id < 0 or block_id >= num_blocks:
            return
        row = kv_layer.view(num_blocks, -1)[block_id]
        n_bytes = int(row.numel()) * int(row.element_size())
        if len(page) != n_bytes:
            return  # size mismatch — leave the block to be recomputed
        raw = torch.frombuffer(bytearray(page), dtype=torch.uint8)
        viewed = raw.view(row.dtype).view_as(row)
        row.copy_(viewed)
    except Exception:
        return


def _bundle_pages(pages: dict[str, bytes]) -> bytes:
    """[u32 num_layers][per layer: u32 name_len][name][u32 page_len][page]]."""
    out = bytearray()
    out += struct.pack(">I", len(pages))
    for name, page in pages.items():
        enc = name.encode("utf-8")
        out += struct.pack(">I", len(enc)) + enc + struct.pack(">I", len(page)) + page
    return bytes(out)


def _unbundle_pages(payload: bytes) -> dict[str, bytes]:
    pages: dict[str, bytes] = {}
    try:
        offset = 0

        def read_u32() -> int:
            nonlocal offset
            value = struct.unpack(">I", payload[offset:offset + 4])[0]
            offset += 4
            return value

        if len(payload) < 4:
            return {}
        num_layers = read_u32()
        for _ in range(num_layers):
            name_len = read_u32()
            name = payload[offset:offset + name_len].decode("utf-8")
            offset += name_len
            page_len = read_u32()
            page = payload[offset:offset + page_len]
            offset += page_len
            if len(page) != page_len:
                return {}
            pages[name] = page
        if offset != len(payload):
            return {}
        return pages
    except Exception:
        return {}


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b if b else 0


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


class SapwoodWorker:
    """Worker-side helper: keeps the paged KV tensors, serializes computed
    blocks to the sapwood server, and rehydrates cached blocks into the paged
    buffer on load.

    Unverified: not run against a real engine.
    """

    def __init__(self, vllm_config, kv_cache_config, server_config: dict[str, Any]):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.server_config = server_config
        self.model_id = server_config["model_id"]
        self.kind = server_config["kind"]
        self._client = KVCacheClient(
            host=server_config["host"],
            port=server_config["port"],
            timeout=server_config["timeout"],
        )
        # layer_name -> torch.Tensor (paged KV cache)
        self.kv_caches: dict[str, Any] | None = None
        # layer_name -> group_idx
        self._layer_to_group: dict[str, int] = {}
        _build_layer_to_group(self.kv_cache_config, self._layer_to_group)
        # logical_block_id -> {layer_name: page_bytes}
        self._store_buffers: dict[str, dict[str, bytes]] = {}
        # current-step metadata
        self._pending: SapwoodConnectorMetadata | None = None

    # -- lifecycle ──────────────────────────────────────────────────────────

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self.kv_caches = dict(kv_caches)

    def set_metadata(self, meta: SapwoodConnectorMetadata) -> None:
        # Replace any leftover buffers from the previous step.
        self._store_buffers.clear()
        self._pending = meta

    # -- store path ─────────────────────────────────────────────────────────

    def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any) -> None:
        if self._pending is None or not self._pending.store_plan:
            return
        group_idx = self._layer_to_group.get(layer_name)
        if group_idx is None:
            return
        for entry in self._pending.store_plan:
            gpu_block_id = entry.gpu_block_ids.get(group_idx)
            if gpu_block_id is None:
                continue
            page = _page_bytes(kv_layer, gpu_block_id)
            if page is None:
                continue
            buf = self._store_buffers.setdefault(entry.logical_block_id, {})
            buf[layer_name] = page

    def wait_for_save(self) -> None:
        if self._pending is None or not self._pending.store_plan:
            self._store_buffers.clear()
            return
        for entry in self._pending.store_plan:
            pages = self._store_buffers.pop(entry.logical_block_id, None)
            if not pages:
                continue
            payload = _bundle_pages(pages)
            try:
                self._client.store_block(
                    block_id=entry.logical_block_id,
                    model_id=self.model_id,
                    start_pos=entry.start_pos,
                    kind=entry.kind,
                    token_hash=entry.token_hash,
                    token_count=entry.token_count,
                    # Blocks are content-addressed, not session-scoped, so a
                    # session_id is not passed to the server. The session is
                    # bookkept client-side via the retrieval path.
                    session_id=None,
                    payload=payload,
                )
            except RuntimeError:
                # Never crash the engine for a cache-store failure; a later
                # resume simply misses the block and recomputes.
                continue
        self._store_buffers.clear()
        self._pending = None

    # -- load path ──────────────────────────────────────────────────────────

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        if self._pending is None or not self._pending.load_plan:
            return
        if self.kv_caches is None:
            return
        for entry in self._pending.load_plan:
            try:
                meta, payload = self._client.load_block(entry.logical_block_id, self.model_id)
            except RuntimeError:
                continue
            if not payload:
                continue
            pages = _unbundle_pages(payload)
            if not pages:
                continue
            for layer_name, page in pages.items():
                group_idx = self._layer_to_group.get(layer_name)
                if group_idx is None:
                    continue
                gpu_block_id = entry.gpu_block_ids.get(group_idx)
                if gpu_block_id is None:
                    continue
                kv_layer = self.kv_caches.get(layer_name)
                if kv_layer is None:
                    continue
                _write_page(kv_layer, gpu_block_id, page)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Loads complete synchronously in this connector.
        return

    def shutdown(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


def _build_layer_to_group(kv_cache_config, out: dict[str, int]) -> None:
    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
    for group_idx, group in enumerate(groups):
        layer_names = getattr(group, "layer_names", None) or []
        for name in layer_names:
            out[name] = group_idx


# ---------------------------------------------------------------------------
# Scheduler side
# ---------------------------------------------------------------------------


def _group_blocks(blocks, group_idx: int) -> list:
    """Return the (block-ish) entries for one KV cache group."""
    try:
        groups = getattr(blocks, "blocks", None)
        if groups is None:
            return list(blocks) if isinstance(blocks, (list, tuple)) else []
        if group_idx < len(groups):
            return list(groups[group_idx] or [])
        return []
    except Exception:
        return []


def _all_groups_blocks(blocks) -> list[tuple[int, Any]]:
    try:
        groups = getattr(blocks, "blocks", None)
        if groups is None:
            return list(enumerate(blocks)) if isinstance(blocks, (list, tuple)) else []
        out: list[tuple[int, Any]] = []
        for group_idx, group in enumerate(groups):
            for block in group or []:
                out.append((group_idx, block))
        return out
    except Exception:
        return []


class SapwoodScheduler:
    """Scheduler-side helper: decides which blocks are stored (computed this
    request touches) and which are loaded (resume). Unverified."""

    def __init__(self, vllm_config, kv_cache_config, server_config: dict[str, Any]):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.server_config = server_config
        self.model_id = server_config["model_id"]
        self.kind = server_config["kind"]
        self._client = KVCacheClient(
            host=server_config["host"],
            port=server_config["port"],
            timeout=server_config["timeout"],
        )
        # request_id -> session_id
        self._sessions: dict[str, str] = {}
        # request_id -> set of logical_block_ids already stored
        self._stored_ids: dict[str, set[str]] = {}
        # request_id -> list of newly planned blocks (accumulated per step)
        self._pending_stores: dict[str, list[SapwoodStorePlan]] = {}
        self._pending_loads: dict[str, list[SapwoodLoadPlan]] = {}

    # -- session bookkeeping ────────────────────────────────────────────────

    def _session_for(self, request) -> str:
        params = getattr(request, "kv_transfer_params", None) or {}
        sid = params.get("session_id") or request.request_id
        self._sessions[request.request_id] = sid
        return sid

    def _session(self, request) -> dict | None:
        sid = self._sessions.get(request.request_id)
        if not sid:
            return None
        try:
            return self._client.get_session(sid)
        except RuntimeError:
            return None

    # -- block size discovery ───────────────────────────────────────────────

    def _block_size(self) -> int:
        groups = getattr(self.kv_cache_config, "kv_cache_groups", None) or []
        for group in groups:
            spec = getattr(group, "kv_cache_spec", None)
            size = getattr(spec, "block_size", None)
            if size:
                try:
                    return int(size)
                except (TypeError, ValueError):
                    continue
        return 16  # best-effort fallback; store still proceeds

    # -- store bookkeeping ──────────────────────────────────────────────────

    def _build_store_plans(self, request, blocks) -> None:
        """Accumulate a store plan for the blocks in *blocks* allocated for
        this request. Idempotent: the server content-addresses blocks."""
        block_size = self._block_size()
        stored = self._stored_ids.setdefault(request.request_id, set())
        # ordinal -> per-group gpu_block_ids, and token_hash (first-seen).
        by_ordinal_gpu: dict[int, dict[int, int]] = {}
        by_ordinal_hash: dict[int, str] = {}
        for group_idx, block in _all_groups_blocks(blocks):
            ordinal = _ordinal_in_group(blocks, group_idx, block)
            if ordinal is None:
                continue
            if getattr(block, "is_null", False):
                continue
            gpu_id = getattr(block, "block_id", None)
            if gpu_id is None:
                continue
            gpu_id = int(gpu_id)
            if gpu_id == 0:
                continue
            hash_val = getattr(block, "block_hash", None)
            if hash_val is None:
                continue  # cannot compute the logical identity honestly
            by_ordinal_gpu.setdefault(ordinal, {})[group_idx] = gpu_id
            if by_ordinal_hash.get(ordinal) is None:
                by_ordinal_hash[ordinal] = str(hash_val)
        plans: list[SapwoodStorePlan] = []
        for ordinal in sorted(by_ordinal_gpu):
            token_hash = by_ordinal_hash.get(ordinal)
            if token_hash is None:
                continue
            start_pos = ordinal * block_size
            logical = block_id_for(self.model_id, start_pos, token_hash, self.kind)
            if logical in stored:
                continue
            stored.add(logical)
            plans.append(
                SapwoodStorePlan(
                    req_id=request.request_id,
                    logical_block_id=logical,
                    start_pos=start_pos,
                    token_hash=token_hash,
                    token_count=block_size,
                    kind=self.kind,
                    gpu_block_ids=dict(by_ordinal_gpu[ordinal]),
                )
            )
        self._pending_stores.setdefault(request.request_id, []).extend(plans)

    # -- load bookkeeping ───────────────────────────────────────────────────

    def _build_load_plans(self, request, blocks, num_external_tokens: int) -> None:
        if num_external_tokens <= 0:
            return
        block_size = self._block_size()
        session = self._session(request)
        if not session:
            return
        stored_by_start: dict[int, dict] = {}
        for b in session.get("blocks", []) or []:
            try:
                start = int(b.get("start_pos", 0))
                if start not in stored_by_start:
                    stored_by_start[start] = b
            except (TypeError, ValueError):
                continue
        if not stored_by_start:
            return
        computed = int(getattr(request, "num_computed_tokens", 0) or 0)
        external_end = computed + int(num_external_tokens)
        # per ordinal -> per-group gpu_block_ids
        by_ordinal: dict[int, dict[int, int]] = {}
        for group_idx, block in _all_groups_blocks(blocks):
            ordinal = _ordinal_in_group(blocks, group_idx, block)
            if ordinal is None:
                continue
            if getattr(block, "is_null", False):
                continue
            gpu_id = getattr(block, "block_id", None)
            if gpu_id is None:
                continue
            gpu_id = int(gpu_id)
            if gpu_id == 0:
                continue
            start = ordinal * block_size
            end = start + block_size
            if end <= computed or start >= external_end:
                continue
            by_ordinal.setdefault(ordinal, {})[group_idx] = gpu_id
        plans: list[SapwoodLoadPlan] = []
        for ordinal in sorted(by_ordinal):
            start = ordinal * block_size
            stored = stored_by_start.get(start)
            if stored is None:
                continue
            plans.append(
                SapwoodLoadPlan(
                    req_id=request.request_id,
                    logical_block_id=str(stored.get("block_id", "")),
                    start_pos=start,
                    token_count=int(stored.get("token_count", block_size)) or block_size,
                    gpu_block_ids=dict(by_ordinal[ordinal]),
                )
            )
        if plans:
            self._pending_loads.setdefault(request.request_id, []).extend(plans)

    # -- hooks ──────────────────────────────────────────────────────────────

    def on_new_request(self, request) -> None:
        self._session_for(request)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """Return (tokens the server can provide beyond num_computed_tokens,
        load_kv_async=False)."""
        session = self._session(request)
        if not session:
            return 0, False
        computed = int(num_computed_tokens)
        blocks = session.get("blocks", []) or []
        total = 0
        for b in blocks:
            try:
                start = int(b.get("start_pos", 0))
                count = int(b.get("token_count", 0))
            except (TypeError, ValueError):
                continue
            end = start + count
            if end <= computed:
                continue  # already locally computed
            if start < computed:
                continue  # partial overlap — conservatively skip
            if start > computed:
                break  # gap — non-contiguous; stop
            total += count
            computed = end
        return int(total), False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self._build_store_plans(request, blocks)
        self._build_load_plans(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output):
        store_plans: list[SapwoodStorePlan] = []
        load_plans: list[SapwoodLoadPlan] = []
        for plans in self._pending_stores.values():
            store_plans.extend(plans)
        for plans in self._pending_loads.values():
            load_plans.extend(plans)
        self._pending_stores.clear()
        self._pending_loads.clear()
        return SapwoodConnectorMetadata(store_plan=store_plans, load_plan=load_plans)

    def request_finished(self, request, block_ids):
        # No blocking work here; stores happen worker-side during execution.
        return False, None


def _ordinal_in_group(blocks, group_idx: int, block) -> int | None:
    """Return the ordinal (index) of *block* within its group list."""
    try:
        group = _group_blocks(blocks, group_idx)
        for i, candidate in enumerate(group):
            if candidate is block or (
                getattr(candidate, "block_id", None) == getattr(block, "block_id", None)
                and getattr(candidate, "block_hash", None) == getattr(block, "block_hash", None)
            ):
                return i
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The connector (vLLM-facing)
# ---------------------------------------------------------------------------


if _VLLM_AVAILABLE:

    class SapwoodKVConnector(KVConnectorBase_V1):  # type: ignore[misc]
        """vLLM KV connector — store computed blocks to sapwood; load cached
        blocks on resume.

        UNVERIFIED: not run against a real engine. See module docstring.
        """

        def __init__(self, vllm_config, role, kv_cache_config):
            super().__init__(vllm_config, role, kv_cache_config)
            server_config = _read_server_config(vllm_config)
            if role == KVConnectorRole.SCHEDULER:
                self.connector_scheduler = SapwoodScheduler(vllm_config, kv_cache_config, server_config)
                self.connector_worker = None
            else:
                self.connector_worker = SapwoodWorker(vllm_config, kv_cache_config, server_config)
                self.connector_scheduler = None

        # -- Worker side --

        def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
            if self.connector_worker is not None:
                self.connector_worker.register_kv_caches(kv_caches)

        def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
            if self.connector_worker is None:
                return
            meta = self._connector_metadata
            if isinstance(meta, SapwoodConnectorMetadata):
                self.connector_worker.set_metadata(meta)
                self.connector_worker.start_load_kv(forward_context)

        def wait_for_layer_load(self, layer_name: str) -> None:
            if self.connector_worker is not None:
                self.connector_worker.wait_for_layer_load(layer_name)

        def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any) -> None:
            if self.connector_worker is not None:
                self.connector_worker.save_kv_layer(layer_name, kv_layer, attn_metadata)

        def wait_for_save(self) -> None:
            if self.connector_worker is not None:
                self.connector_worker.wait_for_save()

        def shutdown(self) -> None:
            if self.connector_worker is not None:
                self.connector_worker.shutdown()

        # -- Scheduler side --

        def on_new_request(self, request) -> None:
            if self.connector_scheduler is not None:
                self.connector_scheduler.on_new_request(request)

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            if self.connector_scheduler is not None:
                return self.connector_scheduler.get_num_new_matched_tokens(
                    request, num_computed_tokens
                )
            return 0, False

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            if self.connector_scheduler is not None:
                self.connector_scheduler.update_state_after_alloc(
                    request, blocks, num_external_tokens
                )

        def build_connector_meta(self, scheduler_output):
            if self.connector_scheduler is not None:
                return self.connector_scheduler.build_connector_meta(scheduler_output)
            return SapwoodConnectorMetadata()

        def request_finished(self, request, block_ids):
            if self.connector_scheduler is not None:
                return self.connector_scheduler.request_finished(request, block_ids)
            return False, None

else:  # vLLM absent

    class SapwoodKVConnector(object):  # type: ignore[no-redef]
        """Placeholder — see module docstring. vLLM is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "sapwood.vllm_adapter requires vLLM; it is not installed."
            )
