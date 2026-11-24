# sapwood-kv

A **network-based disk KV cache for vLLM**. Its purpose: keep long conversations
(and their KV blocks) alive across process boundaries so a session can be
**resumed without re-running the whole prefill** — and so several long
conversations can live under one model's limited GPU memory by keeping only a
hot window on-device while the rest lives in a shared disk store.

Built from scratch on vLLM's native KV-connector API. 

## Goal

> Extend context on a single model and enable sessions to be resumed without
> having to re-compute all the context.

Two halves, honestly weighted:

1. **Session resume** (the core, fully working). A session's KV blocks are
   written to a disk store that lives behind a network protocol. On resume,
   the stored blocks are loaded back — only the blocks the policy selects and
   that are missing need recompute — instead of re-running the entire
   long prefill.
2. **Long / RAG-style context under limited VRAM** (the design intent,
   realized at the block level). The whole conversation lives on disk; only the
   hot window — pinned anchors + the most relevant blocks selected each turn —
   is loaded into GPU memory. Concurrent conversations therefore don't all
   need to be resident in VRAM. This is retrieval-augmented at the block level,
   not a literal extension of the model's positional context window — see
   "What this is not".

## What this is not

- **Not a literal extension of the model's max context window.** Going past a
  model's built-in limit (e.g. 128k) needs RoPE scaling, sliding-window
  attention or retrieval-augmented context — a separate, harder track. Here,
  a long conversation fits *within* the window while only the hot blocks are
  GPU-resident; beyond that you'd add a retrieval policy and a window-size
  budget explicitly.
- **Not a session-fork or branching store.** Sessions are append-only here.
- **Not yet verified end-to-end against vLLM.** Until the
  `docs/CONTAINER-RUNBOOK.md` checklist is green, the adapter is a
  best-effort, single-step, synchronous bridge — not a production offloading
  implementation. The runnable pieces are
  storage, protocol, server, and the resume manager (all covered by tests).
  The vLLM connector subclass (`src/sapwood/vllm_adapter.py`) is wired
  against the real vLLM 0.34.0rc1 `KVConnectorBase_V1` API — scheduler-side
  store/load planning and a worker-side page-serialization/rehydration path —
  with the store/load round-trip covered by tests. It has been run **only**
  against that mocked harness, not against a real engine or the actual model.
  Do not claim it works until it has been exercised in the target container.
- **Valkey backend is not verified against a live server.** The backend
  logic and the stdlib RESP2 framing are covered by tests (in-process fake /
  socketpair), but no `valkey-server` was run here — run one for a real deploy.

## Target model: Inkling 2 Mini

The intended target is **Inkling 2 Mini** served by vLLM (see
[model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini)). Verified
architecture facts (from the model card's `config.json`):

- **Hybrid attention**: 20 sliding-window (local) layers with `sliding_window_size=1024`
  (`local_layer_ids` = 0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22) and
  4 global layers (5, 11, 17, 23) with no window (full 1M context).
- **Short convolution**: each layer owns a short-conv state (`sconv_kernel_size=4`)
  registered in vLLM as an ordinary `SlidingWindowSpec` KV cache — nothing
  special to special-case in the disk cache; it is stored as an ordinary
  KV-cache layer.
- **Sparse MoE**: `n_routed_experts=128`, `num_experts_per_tok=8` — per-token routing;
  irrelevant to block-level KV storage. The disk cache stores KV bytes, not
  expert indices.
- **GQA**: `num_attention_heads=24`, `num_key_value_heads=8`, `head_dim=128`.
- **Looped core**: `loop_config = {core_start=6, core_end=18, num_passes=3, gate_rank=64, is_varloop=false}`.
  The checkpoint (`thinkingmachines/Inkling-2-Mini`, `model_type: inkling2_loop`)
  is a *looped transformer*: the middle 12 layers run 3 passes with shared
  weights, giving **48 layer applications** per token from 24 layers of weights.
  **vLLM 0.34.0rc1 has no loop implementation** — `vllm/models/inkling/`
  iterates `num_hidden_layers` (24) once and the `inkling2_loop` architecture is
  **not in vLLM's registry** (no file in the repo matches `inkling2`). The
  checkpoint falls back to vLLM's transformers backend. The reference
  implementation (`transformers/models/inkling2_loop`) runs the looped core
  once per pass with **deep-copied per-pass KV cache layers** (24 + 2×12 = 48
  cache entries) and re-receives messages between passes via attention-residual
  `block_states`. The KV-cache layer layout the adapter stores is therefore
  **unverified in vLLM** — treat it as correct for the architectures vLLM
  already supports, and resolve the loop question in the container before
  claiming the disk cache is correct for this checkpoint (see
  `docs/CONTAINER-RUNBOOK.md`, section 0).

The KV cache here operates at the **block level** (raw page bytes + a stable
logical id). It is architecture-agnostic and stores one bundle per logical
block (one page per registered layer), so a single block is directly
rehydratable into the engine's paged buffer. Confirmed on-by-construction but
**not yet run end-to-end**:

- the exact `--kv-transfer-config` / connector entry point (verified for
  vLLM **0.34.0rc1** — see `src/sapwood/vllm_adapter.py` docstring),
- the block size in tokens vLLM uses (`config.block_size`; read from
  `kv_cache_config.kv_cache_groups[*].kv_cache_spec.block_size`),
- the exact semantics of the looped core (not visible in vLLM's model code).

## Architecture

```
            ┌────────────────────── one host ──────────────────────┐
            │                                                      │
   node A │  vLLM (Inkling 2 Mini)          node B │ vLLM ...    │
            │     │                                              │
            │     ▼                                              │
            │  SapwoodKVConnector  (thin shim; see vllm_adapter) │
            │     │                                              │
            │     ▼  framed TCP (JSON control + binary blocks)    │
            │                                                     │
            │  ┌───────────────────────┐                         │
            │  │  sapwood-kv-server   │   cache-dir             │
            │  │  (CacheServer)        │ ───► blocks/ + sessions/│
            │  └───────────────────────┘     (content-addressed) │
            │                                                     │
            └──────────────────────────────────────────────────────┘
```

- **Disk store** (`storage.py`) — content-addressed blocks + a per-session
  index, atomic writes. One server owns it.
- **Network protocol** (`protocol.py`) — length-prefixed frames: JSON header
  for control + binary payload for blocks. No base64 overhead.
- **Server** (`server.py`) — `sapwood-kv-server`; one thread per client; owns
  the disk store and the session index.
- **Resume manager** (`connector.py`) — protocol-backed store/load/resume.
- **RAG-style selection** (`retrieval.py`) — decides which blocks enter the
  hot GPU window (recency + pinned anchors; semantic scorer is a pluggable
  hook).
- **vLLM connector** (`vllm_adapter.py`) — a `KVConnectorBase_V1`
  subclass wired against the real vLLM 0.34.0rc1 API (scheduler + worker
  sides, store + load paths). **UNVERIFIED**: never run against a real
  engine — see the module docstring.

## Quick start (this repository)

The repository ships with no dependencies beyond a Python 3.10+; the server
uses stdlib only. Dev tooling uses pytest.

```bash
cd sapwood

# create an isolated venv (recommended — the system Python blocks pip writes)
python3 -m venv .venv
source .venv/bin/activate

# install editable + dev extras
python3 -m pip install -e ".[dev]"

# one-shot smoke: start the server, run resume against it
sapwood-kv-server --host 127.0.0.1 --port 7676 --cache-dir .sapwood-cache &
```

Then, from any client (or the vLLM connector), talk to the server:

```python
from sapwood import KVCacheClient, KVResumeManager, BlockSelectionPolicy
from sapwood.connector import build_default_manager

with KVCacheClient("127.0.0.1", 7676) as client:
    session = client.create_session("Inkling-2-Mini", title="long-chat")
    sid = session["session_id"]

    # store a computed block (payload = opaque KV bytes)
    client.store_block(block_id=..., model_id="Inkling-2-Mini", start_pos=0,
                       kind="prefix", token_hash="<hash>", token_count=16,
                       session_id=sid, payload=b"...")

    # resume without recomputing: load selected blocks from disk
    mgr = KVResumeManager(client)
    hot = mgr.resume_session(sid)   # list of (meta, payload), oldest first
```

See `tests/test_protocol.py` for a complete worked example.

## Storage backends

The server owns the store and exposes one framed TCP interface, so swapping the
backend is invisible to clients. Two are supported:

| Backend | Where | Notes |
| --- | --- | --- |
| `disk` (default) | local files under `--cache-dir` | 100% stdlib, atomic temp+rename writes |
| `valkey` | a Valkey server over TCP (`--valkey-url`) | remote / shared across nodes |

No third-party client is needed for either — the Valkey backend speaks RESP2
through a minimal stdlib client (`valkey_client.py`), so the package stays
dependency-free and US-origin. Valkey itself is a Linux Foundation project
(BSD-3-Clause) — not a Chinese-origin dependency.

```bash
# remote, shared across nodes
sapwood-kv-server --backend valkey --valkey-url redis://127.0.0.1:6379/0

# one server, one disk (the default)
sapwood-kv-server --cache-dir .sapwood-cache
```

Implementation notes / limits:

- **Keys**: `sb:blk:<model_slug>:<block_id>` (one JSON doc with the base64
  payload + meta) and `sb:sess:<session_id>` (session descriptor).
- **Values are single keys** — a block's meta + binary payload round-trip in one
  `GET`.
- **Session listing** scans `sb:sess:*` and filters — fine for small-to-moderate
  session counts, not designed for 100k+ sessions.
- **Concurrency**: single writer per session (`append_block` is
  read-modify-write), same assumption as the disk backend; `create` uses an
  atomic `SET NX`. Run one `sapwood-kv-server` per Valkey database.
- The optional `valkey`/`redis` PyPI client is *not* used — the RESP2 client is
  stdlib-only. If you'd rather use the official client, pass a `client` object
  (anything with `set`/`get`/`delete`/`exists`/`keys`) to
  `sapwood.valkey_backend.make_backend(config, client=...)`.
- Live-server end-to-end (vLLM -> server -> Valkey -> resume) is **not**
  verified in this environment — the backend logic is covered by tests with an
  in-process fake client, but a real Valkey server must be run for an actual
  deploy.

## Retrieval-assembly (hot window)

The RAG-style path decides *which* blocks enter the GPU window each turn:

```python
from sapwood import KVResumeManager, BlockSelectionPolicy, assemble_window
from sapwood.connector import build_default_manager

mgr = build_default_manager("127.0.0.1", 7676, "Inkling-2-Mini", max_hot_blocks=4)

# 1) assemble the window (e.g. a 256K-token budget, 1M conversation):
selected, summary = mgr.assemble_context(
    session_id, query_tokens=[...],     # token IDs of the current query
    max_blocks=4, max_tokens=65536,      # e.g. ~64K tokens of context
    pinned=["b_anchor_system_prompt"],  # anchors always kept
)

# 2) load the payloads for those blocks
hot = mgr.resume_session(session_id, query_tokens=[...])
for meta, payload in hot:
    ...  # feed to vLLM / restore KV
```

`assemble_context` (and `resume_session(query_tokens=...)`) scores every stored
block by **lexical relevance to the query** (token-overlap Jaccard) plus a
**recency bonus**, force-includes pinned anchors, then greedily fills the
budget and returns the blocks ordered **oldest-first** so the model sees a
coherent, ordered context.

- **Relevance is lexical, not semantic.** A semantic (embedding-based) scorer
  can be plugged in via the `scorer=` argument to `assemble_window` / the
  `BlockSelectionPolicy` hook — it is not bundled here.
- **Anchors are additive, not capped** — pinned blocks are always kept and do
  not count toward the `max_blocks` budget.
- **Budget** — `max_blocks` (hard block cap) and/or `max_tokens` (hard token
  cap). The whole conversation (even 1M tokens) lives on disk; only the window
  is GPU-resident.

Every block stores its token IDs in the meta sidecar (persisted on `store`),
which is what makes the lexical scoring local and cheap.

## CacheBlend-style selective recompute

When an assembled window is replayed into the model, the cached KV for a block is
computed under a prefix that included the blocks we skipped. The window is not a
perfect contiguous prefix replay, so blocks that follow a gap are stale.

The CacheBlend pattern recovers quality cheaply: recompute only the most-affected
blocks and reuse the rest as-is. `sapwood.cacheblend.compute_reuse_plan` decides
this over the assembled window:

- **Staleness** is a structural proxy (no model, no GPU): a kept block is stale
  when its immediate left neighbor in the window does not end exactly where it
  starts — the gap is the skipped context that block's hidden states were
  computed against. The **first** kept block is the assumed base of the replayed
  context (staleness 0); pinned anchors are trusted (staleness 0) and never recomputed.
- **Budget**: recompute the most-stale blocks until the recompute-token budget
  (`recompute_fraction` × the window's total tokens, default **0.15** in
  `ClientConfig`) is exhausted. At least the single most-stale block is
  recomputed when any are stale (a block is the atomic unit here — the original
  CacheBlend paper sub-selects tokens, which needs attention statistics the
  model must supply). Set `recompute_fraction=0` to disable recompute entirely.
- **Returns** a `ReusePlan` (`reusable` / `recompute` lists + `staleness` +
  token accounting) — the engine does the actual recompute; this layer only decides.

```python
from sapwood import KVResumeManager
from sapwood.cacheblend import reuse_plan_summary

mgr = build_default_manager("127.0.0.1", 7676, "Inkling-2-Mini", max_hot_blocks=4)
selected, summary, plan = mgr.assemble_window_with_reuse(session_id, query_tokens=[...])
print(reuse_plan_summary(plan))

# reusable blocks  -> load payloads via mgr.resume_session / load_block
# recompute blocks -> re-run prefill for them (vLLM-side)
```

`KVResumeManager.recompute_blocks(session_id, ...)` is the one-call convenience
that returns just the blocks needing recompute.

Honestly, what it is not: it is not a token-level quality-recovery algorithm and
it does not compute attention. The engine-level hook for real recompute decisions
(top-K tokens by attention deviation, or a model-supplied staleness fn) is the
`staleness_fn` argument to `compute_reuse_plan` — plug it in where you have the
model; this layer stays model-free and fully testable.

## Running the tests

```bash
. .venv/bin/activate
python3 -m pytest -q
```

All 76 tests pass (storage, protocol round-trip over a real socket, server
end-to-end, retrieval selection).

## Configuration

| Setting | Where | Notes |
| --- | --- | --- |
| `--host` / `--port` | `sapwood-kv-server` | protocol endpoint |
| `--cache-dir` | `sapwood-kv-server` | disk store root (default `.sapwood-cache`) |
| `--backend` | `sapwood-kv-server` | `disk` (default) or `valkey` |
| `--valkey-url` | `sapwood-kv-server` | Valkey URL when `--backend=valkey` (default `redis://127.0.0.1:6379/0`) |
| `block_size` (tokens) | `config.ClientConfig` | must match vLLM's KV block size |
| `max_hot_blocks` | `config.ClientConfig` | hot window size in GPU |
| `pinned_blocks` | `ClientConfig` / `BlockSelectionPolicy` | block ids kept in the window regardless of recency |
| `recompute_fraction` | `config.ClientConfig` | CacheBlend selective-recompute budget (default 0.15) |

## Project layout

```
sapwood/
├── pyproject.toml            # package metadata (src layout), console script
├── LICENSE                   # MIT
├── README.md
├── src/sapwood/
│   ├── __init__.py           # public API
│   ├── storage.py            # content-addressed BlockStore + SessionIndex
│   ├── protocol.py           # framed TCP protocol (KVCacheClient + helpers)
│   ├── server.py             # CacheService + CacheServer + CLI entry
│   ├── retrieval.py          # BlockSelectionPolicy (RAG-style hot window)
│   ├── cacheblend.py        # selective-recompute planning (CacheBlend-style)
│   ├── valkey_backend.py    # remote Valkey storage (remote/shared)
│   ├── valkey_client.py     # minimal stdlib RESP2 client
│   ├── connector.py          # KVResumeManager (resume/store/lifecycle)
│   ├── config.py             # ServerConfig / ClientConfig
│   └── vllm_adapter.py       # vLLM shim (UNVERIFIED, see file docstring)
└── tests/
    ├── test_storage.py       # store + session index
    ├── test_protocol.py      # framing + real-socket end-to-end
    ├── test_retrieval.py     # block selection policy
    ├── test_assembly.py      # window assembly + lexical scoring
    ├── test_cacheblend.py    # selective-recompute planning
    └── test_valkey.py        # Valkey backend + RESP2 framing
```

## On going beyond the max context window

If you want a conversation *longer* than the model's native window, that
requires a different approach than a disk KV cache: sliding-window, RoPE
scaling, or a retrieval layer that chooses a window of relevant blocks.
`BlockSelectionPolicy.scorer` is the hook for plugging a semantic selector in;
the disk store and resume path are unaffected. For the *quality recovery*
side of reusing a window with gaps, the CacheBlend-style layer in
`cacheblend.py` (see above) is the hook — it is a planning layer; the
engine (vLLM) does the actual recompute, ideally with a model-supplied
`staleness_fn`.

## Roadmap / open questions

1. ~~Wire `vllm_adapter.py`~~ — done against vLLM 0.34.0rc1 (scheduler/
   worker sides, store/load paths). Still **unverified** — needs an actual
   engine run against the Inkling 2 Mini model in the target container
   (see `docs/CONTAINER-RUNBOOK.md` for the runbook, with the looped-model
   mismatch flagged as the first gate).
2. Plug a real semantic relevance scorer into the assembly path (`scorer=`
   argument to `assemble_window` / `BlockSelectionPolicy.scorer`) — the
   current relevance is lexical only.
3. Serializer for the model's KV layout so stored bytes are directly
   loadable by vLLM (currently opaque payloads).
4. Multi-process / multi-owner safety for the disk store (today: single
   server owns it).
5. Branching/forking sessions, and TTL-based eviction.
6. System-prompt/anchor pinning via stable ids (currently by block id).

## License

MIT (see `LICENSE`).
