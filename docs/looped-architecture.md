# Looped-architecture notes — Inkling 2 Mini under vLLM

Status: **design analysis from source inspection — not verified against a
running engine.** Nothing on this page has been executed in the target
container. The first thing to resolve in the container is exactly what vLLM
does with this checkpoint (see `docs/CONTAINER-RUNBOOK.md`, §0).

## 1. The checkpoint

`thinkingmachines/Inkling-2-Mini` (`config.json`):

- Top-level: `architectures: ["Inkling2LoopForConditionalGeneration"]`,
  `model_type: "inkling2_loop"`.
- Nested `text_config`: `model_type: "inkling2_loop_text"`, `num_hidden_layers: 24`,
  `model_max_length: 1048576`, `hidden_size: 2304`, `num_attention_heads: 24`,
  `num_key_value_heads: 8`, `head_dim: 128`, `d_rel: 16`, `rel_extent: 1024`,
  `local_layer_ids: [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22]`
  (20 local SWA layers; global = 5, 11, 17, 23), `sliding_window_size: 1024`,
  `swa_head_dim: 128`, `swa_num_attention_heads: 24`, `swa_num_key_value_heads: 8`,
  `sconv_kernel_size: 4`, `n_layers_per_attn_res_block: 6`,
  `attn_res_normalize_values: true`, `n_routed_experts: 128`,
  `num_experts_per_tok: 8`, `intermediate_size: 1152`, `mlp_gate_softcap: 4.0`,
  `mlp_up_softcap: 25.0`, `route_scale: 8.0`, `use_gate_bias: true`,
  `log_scaling_n_floor: 8192`, `log_scaling_alpha: 0.1`,
  `final_logit_softcapping: 30.0`,
  `local_layer_ids`/`sconv_kernel_size`/`rel_extent` etc.
- **`loop_config: {core_start: 6, core_end: 18, num_passes: 3, gate_rank: 64, is_varloop: false}`** —
  inside `text_config`.
- Model card: 24-layer decoder-only, sparse MoE — **the middle 12 layers run
  three times with shared weights, giving 48 layer applications per token
  from 24 layers of weights**. ~26.4B total, ~3.4B active per token (5.4B
  with the loop passes). Native multimodal. 1M context.

## 2. What the reference (HF transformers) implementation does

`src/transformers/models/inkling2_loop/modeling_inkling2_loop.py` —
the checkpoint's own reference architecture:

- Layers `core_start` (6) through `core_end - 1` (17) — the **looped core** —
  are run `num_passes` (3) times.
- Each pass uses **deep-copied per-pass KV cache layers**
  (`cache.layers += [copy.deepcopy(layer) for layer in loop_cache_layers * (num_passes - 1)]`),
  so the checkpoint's KV cache holds **48 entries**: 24 original +
  (3−1)×12 looped-core copies.
- The first pass runs on the pre-computed hidden states; passes 2–3
  **restart from the pre-loop states** and receive **messages from the
  previous pass** — `message_input(previous_block_states, previous_hidden_states, loop_layer_idx)` —
  where `block_states` are attention-residual messages accumulated across
  passes (`block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)])`
  on `is_block_boundary = layer_idx % n_layers_per_attn_res_block == 0`).
- Cache offset per pass:
  `cache_offset = num_hidden_layers + (loop_idx - 1) * num_loop_layers - loop_start`.
- Message gates (`Inkling2LoopMessageGate`, `gate_rank: 64`),
  attention residuals (`Inkling2LoopMessageAttentionResidual`), and
  dynamic short convolutions (`Inkling2LoopDynamicShortConvolution`) are all
  part of this reference implementation — **none of which exist in vLLM**.

## 3. What vLLM 0.34.0rc1 ships

(`vllm/models/inkling/` at tag `v0.34.0rc1`)

- `InklingModel.__init__` calls
  `make_layers(config.num_hidden_layers, ...)` for a **flat 24-layer** model,
  and the forward loop iterates `self.layers[self.start_layer : self.end_layer]`
  **once**. There is no loop anywhere.
- `loop_config`, `n_layers_per_attn_res_block`, `attn_res_normalize_values` and
  `gate_rank` are not referenced anywhere in the model, config, attention, or
  conv-state code — `loop_config` is silently swallowed by `**kwargs` in
  `InklingModelConfig`.
- `InklingModelConfig` is registered under `model_type: "inkling_model"`, and a
  corresponding `inkling_mm_model`. Neither matches the checkpoint's
  `inkling2_loop_text` / `inkling2_loop`.
- The architecture registry (`vllm/model_executor/models/registry.py`) maps
  only `InklingForCausalLM` and `InklingForConditionalGeneration`
  (`vllm/models/inkling/`). `Inkling2LoopForConditionalGeneration` is **absent**
  — no file in the repo matches `inkling2`.
- The short-conv state is registered as an ordinary paged `SlidingWindowSpec`
  (`InklingConvState`, `block_size = sconv_kernel_size = 4`, packed K/V/attn/mlp
  streams head-major into one page) and consumed out-of-band by a custom
  backend (`InklingSconvBackend` / `InklingSconvMetadataBuilder`). The wrapper
  sets `self.uses_sconv = True` ("Read by the **MRV2 runner** to publish
  per-request short-conv metadata") — the MRV2 runner is the
  `VLLM_USE_V2_MODEL_RUNNER=1` path (see the Inkling serving recipes), and its
  consumption of `uses_sconv` is unverified from this code path (absent in the
  V1 `gpu/model_runner.py`, which has 0 references to `sconv`/`uses_sconv`/`loop`).
- Hybrid attention: local layers (0–4, 6–10, 12–16, 18–22) → `SlidingWindowSpec`
  (`sliding_window = sliding_window_size = 1024`); global layers (5, 11, 17, 23)
  → `FullAttentionSpec`.
- MoE: 128 routed experts, 8 active per token — per-token routing; the disk
  cache stores KV bytes, not expert indices, so this does **not** affect the
  cache.

## 4. How the checkpoint resolves in vLLM

`_CONFIG_REGISTRY` in `vllm/transformers_utils/config.py` only registers
`inkling_mm_model` and `inkling_model`. The top-level `model_type` is
`inkling2_loop` — not registered. Resolution falls back to the transformers
backend (`AutoConfig`/`_try_resolve_transformers`), which hands the engine the
installed `transformers` package's `Inkling2LoopForConditionalGeneration`
(Python source from the same HF repo). Consequences:

- The **native vLLM Inkling implementation cannot serve this checkpoint** — it
  would need to map the checkpoint's architecture to a registered class, and
  that does not happen today.
- The **transformers backend** *does* implement the loop correctly (it is the
  reference implementation), but it is (a) not the optimized path — it bypasses
  the fused qkvr prep, FA4 rel-attention kernels, and NVFP4 handling the
  native path provides; and (b) entirely unverified against sapwood's connector,
  which assumes the standard `static_forward_context` / `register_kv_caches`
  layout that the native `vllm/models/inkling/` classes populate. The
  transformers backend does not populate `static_forward_context` the same way.

## 5. Implications for sapwood

- **Don't assume the engine registers 24 flat layers.** In the looped
  reference, the engine registers 48 cache entries (24 + 2×12), each
  representing (layer, pass). The adapter ships one page per registered layer —
  it must measure, not assume (see §0 of the runbook).
- **The adapter must be confirmed to fire against whatever path the engine
  actually takes.** If the transformers fallback does not route through
  `static_forward_context` / `register_kv_caches`, sapwood would store nothing
  — which is safe (best-effort, never crashes the engine) but means the resume
  path is dead in practice.
- **Cache size and block-count grow ~2× on the looped half.** Looped-core KV
  for 12 layers × 3 passes = 36 entries vs 12 naive — don't underestimate
  disk / transfer cost.

## 6. Decision gate (see `docs/CONTAINER-RUNBOOK.md` §0)

Before the store→resume verification is meaningful, resolve in the container:

1. Which backend does vLLM actually pick for this checkpoint? (`model_impl`
   / startup logs; transformers fallback vs a native path.)
2. How many cache layers does the engine report — 24 or 48 (or another number)?
   (`kv_caches` keys / `kv_cache_config.kv_cache_groups`.)
3. Does sapwood's `register_kv_caches` see those layers?
4. Is the connector's store/load path functional against the actual serving
   path?

If the checkpoint cannot be served by the target container natively, the vLLM
recipe's verified target is `thinkingmachines/Inkling-NVFP4` (975B, 4×
GB200/B200-class) — the native vLLM Inkling path is heavier and untested for
the small Inkling 2 Mini checkpoint. Either the target changes or the loop
question is resolved in-container first.

## 7. References

vLLM (tag `v0.34.0rc1`):

- `vllm/models/inkling/nvidia/model.py` — flat 24-layer `InklingModel`,
  `uses_sconv = True`, `InklingDecoderLayer`, `InklingConvState` wiring.
- `vllm/models/inkling/nvidia/attention.py` — `InklingAttention`,
  `SlidingWindowSpec` / `FullAttentionSpec`, hybrid local/global split.
- `vllm/models/inkling/configs.py` — `InklingModelConfig`, `loop_config`
  swallowed by `**kwargs`, `mamba2_cache_params`.
- `vllm/models/inkling/nvidia/sconv_swa_attn.py` — `InklingConvState` /
  `InklingSconvBackend` (packed 4-stream paged `SlidingWindowSpec`, block 4).
- `vllm/model_executor/models/registry.py` — only `InklingForCausalLM` /
  `InklingForConditionalGeneration` registered.
- `vllm/transformers_utils/config.py` — `_CONFIG_REGISTRY` registers
  `inkling_model` / `inkling_mm_model` only.
- `vllm/v1/worker/gpu/model_runner.py` — enumerates attention layers via
  `compilation_config.static_forward_context`, calls `init_kv_cache`, hands
  `kv_caches_dict` to the connector. 0 references to `sconv`/`uses_sconv`/`loop`.
- `vllm/v1/worker/gpu/model_states/default.py` — `prepare_attn` →
  `build_attn_metadata` (model-specific metadata threaded through
  `model_specific_attn_metadata`).
- `recipes/models/thinkingmachines/Inkling.yaml` — verified serving path:
  `thinkingmachines/Inkling-NVFP4`, `VLLM_USE_V2_MODEL_RUNNER=1`, native Inkling.

HF transformers (reference):

- `src/transformers/models/inkling2_loop/modeling_inkling2_loop.py` — the
  looped implementation: `loop_start`/`loop_end`, `num_passes`,
  per-pass deep-copied cache layers, `message_input` / attention residuals.
- `src/transformers/models/inkling2_loop/configuration_inkling2_loop.py` —
  `loop_config` consumed by `Inkling2LoopConfig`.
- `docs/source/en/model_doc/inkling2_loop.md`.

Checkpoint & model card:

- `thinkingmachines/Inkling-2-Mini` `config.json` — architectures, model types,
  `loop_config`, hybrid SWA/global split.
- Model card (huggingface.co) — "the middle 12 layers run three times with
  shared weights, giving 48 layer applications per token from 24 layers of
  weights"; "Support for looped models in local runtimes is new, and speed
  varies by runtime and hardware."
