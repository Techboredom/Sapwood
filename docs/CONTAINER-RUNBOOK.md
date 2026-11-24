# Sapwood × vLLM container runbook — store→resume verification

Purpose: exercise the sapwood disk KV cache against a real vLLM engine in the
target container image, confirm that computed KV blocks are shipped to the
disk cache, and that a resumed session loads those blocks instead of
re-computing.

**Nothing in this document is verified until the checkboxes at the bottom are
actually checked. The vLLM adapter (`src/sapwood/vllm_adapter.py`) has never
been executed end-to-end. Treat everything below as a procedure, not a
result.**

---

## 0. Critical gate — looped-model architecture mismatch

Before anything else, resolve this. It determines whether the run is even
meaningful.

**Checkpoint reality** (`thinkingmachines/Inkling-2-Mini`, `config.json`):

- Top-level: `architectures: ["Inkling2LoopForConditionalGeneration"]`,
  `model_type: "inkling2_loop"`.
- Nested `text_config`: `model_type: "inkling2_loop_text"`, `num_hidden_layers: 24`,
  `local_layer_ids` (0–4, 6–10, 12–16, 18–22; global = 5, 11, 17, 23),
  `sliding_window_size: 1024`, `sconv_kernel_size: 4`,
  `n_layers_per_attn_res_block: 6`, `attn_res_normalize_values: true`,
  **`loop_config: {core_start: 6, core_end: 18, num_passes: 3, gate_rank: 64, is_varloop: false}`**.

**What the model card says** (https://huggingface.co/thinkingmachines/Inkling-2-Mini):

> 24-layer decoder-only transformer with sparse MoE; **the middle 12 layers run
> three times with shared weights, giving 48 layer applications per token from
> 24 layers of weights**. Native multimodal — images encoded via a vision
> transformer.

**What HF transformers' reference implementation does**
(`src/transformers/models/inkling2_loop/modeling_inkling2_loop.py`):

- Layers 6–17 (the looped core) are run **once per pass**, `loop_num_passes=3`.
- Each pass of the looped core uses **its own deep-copied KV cache layer**
  (`cache.layers += [deepcopy(layer) for layer in loop_cache_layers * (num_passes-1)]`)
  — i.e. the checkpoint's KV cache is **48 layer entries**: 24 original + 2
  extra copies × 12 looped layers.
- Remaining passes **restart from the pre-loop states** and receive messages
  from the previous pass (`message_input` / attention-residual `block_states`,
  `cache_offset = num_hidden_layers + (loop_idx-1)*num_loop_layers - loop_start`).

**What vLLM 0.34.0rc1 ships** (`vllm/models/inkling/nvidia/model.py`,
`attention.py`, `configs.py`, `sconv_swa_attn.py`, `registry.py`,
`transformers_utils/config.py`):

- Registers only `InklingForCausalLM` / `InklingForConditionalGeneration`
  (`vllm/models/inkling/`), a **flat 24-layer** implementation (`make_layers`
  over `num_hidden_layers`, iterated once). **Zero occurrences** of
  `inkling2` / `loop_config` anywhere in the repo.
- The `loop_config` field is silently swallowed by `**kwargs` in
  `InklingModelConfig` and **never consumed**.
- The registry (`vllm/model_executor/models/registry.py`) has no entry for
  `Inkling2LoopForConditionalGeneration`.

**Consequence:** the checkpoint is not loadable by vLLM's native
`vllm.models.inkling` path. vLLM's resolution falls back to the **transformers
backend** (`model_impl` falls back; `_try_resolve_transformers`), which uses
the installed `transformers` package's `Inkling2LoopForConditionalGeneration`
(Python source, looped architecture, deep-copied per-pass cache layers).

**Ask / verify in the container:**

1. `vllm serve <checkpoint>` — does it start, and which backend does it pick?
   (Look for `model_impl` / "Transformers" / transformers in the startup logs;
   check the resolved arch and the KV cache layers in the log.)
2. Once it starts: does the engine report **48 cache layers** (24 + 2×12), or
   24? (`gpu/model_runner.py` → `kv_caches_dict` via
   `compilation_config.static_forward_context` → `register_kv_caches`.)
3. If the transformers backend is used, does `register_kv_caches` see the
   per-pass deep-copied loop layers, or only the 24 original layers?

**Decision gate.** This matters for the disk cache in a direct way: the
adapter ships **one page per registered layer**. If the engine only knows 24
flat layers, the disk cache is "correct" only for a model that was served that
way — but the checkpoint was trained for 48 passes. Either the checkpoint
needs a vLLM-loadable form (e.g. the recipe's verified target
`thinkingmachines/Inkling-NVFP4`), or the adapter must be extended to handle
per-pass loop-cache layers. Do NOT proceed to the store/resume steps below
until this is resolved and documented.

If the container cannot service the checkpoint at all (rejected architecture),
that is already a finding: pick an alternate serving target or a compatible
quantization of this checkpoint. Do not weaken the finding to "looks close".

---

## 1. Prerequisites

- Podman machine started (`podman machine list` shows it running).
- The target container image present:
  `vllm/vllm-openai:nightly-f8b29fab414884206460fe63b983ca34fc6e3e9b`
  (vLLM 0.34.0rc1). Verify: `podman images | grep vllm`.
- A GPU is mounted into the container (do not run on CPU — the model wants
  ≥32GB VRAM; NVFP4 verification recipe targets 4× GB200/B200-class nodes,
  so Inkling 2 Mini (26.4B) should fit a modest GPU but verify).
- Sapwood installed in the **same** environment as vLLM (either the container
  image itself, or a shared mounted volume + a second container).
- The sapwood-kv-server running and reachable from the vLLM worker.

## 2. Install sapwood into the container image

From the repo root (host side):

```bash
# pip install sapwood from this repo into the image:
podman run --rm -v "$PWD:/src:ro" \
    vllm/vllm-openai:nightly-f8b29fab414884206460fe63b983ca34fc6e3e9b \
    pip install /src
```

(Adjust as needed for the container’s Python/pip layout. The install must
land in the environment vLLM actually runs in; the adapter imports
`sapwood.vllm_adapter` from the venv used by the engine.)

## 3. Start sapwood-kv-server

Run the server next to vLLM. It listens on a TCP port and serves the disk
cache:

```bash
podman run -d --name sapwood-kv --network host \
    vllm/vllm-openai:nightly-f8b29fab414884206460fe63b983ca34fc6e3e9b \
    sapwood-kv-server --host 127.0.0.1 --port 7676 --cache-dir /data/kv-cache
```

Verify:

```bash
podman logs sapwood-kv | grep -i -E "listening|serving|ready"
```

## 4. Launch vLLM with the connector

Attach the connector via `--kv-transfer-config`. The server must be reachable
at the configured host/port (from the worker’s network namespace):

```bash
podman run -d --name vllm --network host \
    --gpus all \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    vllm/vllm-openai:nightly-f8b29fab414884206460fe63b983ca34fc6e3e9b \
    vllm serve thinkingmachines/Inkling-2-Mini \
        --kv-transfer-config '{
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
```

Notes:

- `VLLM_USE_V2_MODEL_RUNNER=1` — the Inkling serving recipes set this; it is
  the MRV2 path that publishes per-request short-conv metadata. Keep it set.
- `--network host` (or a shared network + the host IP) so the worker can reach
  `127.0.0.1:7676`.
- The adapter is registered in `sapwood.vllm_adapter`; vLLM loads it lazily
  from the installed package. It needs **no** in-repo wiring.

## 5. Verification steps

Each of these is a real check against the live engine. Until each is green,
the adapter is unverified.

### 5a. Engine starts without connector import errors

```bash
podman logs vllm 2>&1 | grep -i -E "sapwood|kv_connector|error|Traceback"
```

Expected: connector loaded (`SapwoodConnector`), no import or traceback errors.

### 5b. Both scheduler and worker instantiate the connector

```bash
podman logs vllm 2>&1 | grep -i -E "Sapwood|scheduler|worker"
```

Expected: construction logged for both roles (scheduling side and worker
side). The adapter dispatches on `KVConnectorRole`; a failure here is the
first place to look if the store path never triggers.

### 5c. A request computes blocks and ships them to sapwood

Send an opaque prompt (or a resume-style long conversation; keep it small for
this check so only a short prefix is computed). Then:

```bash
podman logs vllm 2>&1 | grep -i -E "sapwood|store"
# inspect the disk cache directory on the host / shared volume:
find /data/kv-cache -type f | head
```

Expected: per-block, per-layer pages written to the disk cache, deduped by
`block_id` (content-addressed → repeated stores of the same block are
idempotent).

### 5d. Resume: the same conversation loads from the cache instead of recomputing

Re-send the same session (same `session_id` via `kv_transfer_params`, or
restart and replay), and watch:

```bash
podman logs vllm 2>&1 | grep -i -E "sapwood|load|match"
```

Expected: `get_num_new_matched_tokens` > 0, `load_block` calls, pages written
back into the paged buffer. **No re-computation of the cached prefix.**

### 5e. Failure path (negative test)

Stop the sapwood server; re-send the request. Expected: the connector
surfaces the failure and **falls back to recompute** — the engine must not
crash. This confirms the best-effort design (never crash the engine on cache
store/load failure).

### 5f. Confirmation loop

Restart the server; re-send the same session. It should load from the disk
cache again. Stop and restart vLLM; repeat. Confirm the resume path is
stable across engine restarts.

## 6. Checklist

- [ ] Checkpoint resolution: arch backend + backend chosen, **48 vs 24 cache
      layers** documented. Either vLLM serves this checkpoint natively or this
      is documented as a blocker.
- [ ] `sapwood-kv-server` reachable from the worker.
- [ ] Engine starts with the connector; no import errors.
- [ ] Both scheduler + worker instances log connector construction.
- [ ] A real request ships blocks to the disk cache (files on disk).
- [ ] A resumed session loads blocks from the cache (no recompute of the
      cached prefix).
- [ ] Server-down negative test: fallback to recompute, no engine crash.
- [ ] Restart stability.

## 7. Known gaps / honest status

- **vLLM adapter has never been run** against a live engine. It is a narrow
  single-step, synchronous, best-effort bridge — not a production offloading
  implementation (no sliding-window/recurrent-state draining, no EAGLE/MTP
  groups, no partial tails, no GPU-block fences, no async stores).
- **The looped-model question is open.** vLLM ships a flat 24-layer
  implementation (`vllm/models/inkling/`) with no loop consumption; the
  checkpoint is designed for 48 pass-applications. The serving path (vLLM
  native vs the transformers fallback) and the resulting KV-cache layer
  layout must be confirmed in the container before the disk cache can be
  presented as correct for Inkling 2 Mini.
- **Retrieval-assembly and CacheBlend** are client-side (`connector.py`,
  `cacheblend.py`); the adapter only moves bytes. The hot-window decision
  lives outside the engine.
- The MTP head (`InklingMTP`) is not handled by this adapter; MTP verification
  is out of scope.
- Torch-gated adapter tests skip in the sandbox (torch not installed); the
  3 skipped tests are the page-extraction/serialization paths, not the
  store→load round trip.

## 8. Appendix — registration path (unverified)

```json
{
  "kv_connector": "SapwoodConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "sapwood.vllm_adapter:SapwoodKVConnector",
  "kv_connector_extra_config": {
    "server_host": "127.0.0.1",
    "server_port": 7676,
    "timeout": 60.0,
    "model_id": "thinkingmachines/Inkling-2-Mini"
  }
}
```
