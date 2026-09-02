# Faithful ShadowKV Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-ml:subagent-driven-development (recommended) or superpowers-ml:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an accuracy-first `attention=shadowkv` implementation that follows ByteDance-Seed/ShadowKV's `ShadowKVCache` selection and reconstruction algorithm while preserving exact dense prefill and vLLM's existing FlashAttention decode path.

**Architecture:** Add a focused `shadowkv.py` attention implementation with explicit per-layer tensors for global pre-RoPE SVD factors, post-RoPE landmarks, outlier chunks, and selected chunks. Extend the existing handler with a narrow pre-RoPE capture/consume interface and patch vLLM's scalar `RotaryEmbedding.__call__` only when ShadowKV is selected; the existing FlashAttention v1 patch remains the execution boundary. Decode appends to vLLM's normal cache, gathers values from that cache, reconstructs only selected keys from local `SV` and shared `U`, applies the captured RoPE implementation at original positions, and invokes vLLM's varlen FlashAttention directly on reusable assembled-token buffers.

**Tech Stack:** Python 3.10+, PyTorch, vLLM 0.11.0, vLLM FlashAttention, Hydra YAML, pytest.

---

### Task 1: Lock the official algorithm and integration contracts in tests

**Files:**
- Create: `tests/test_shadowkv_algorithm.py`
- Create: `tests/test_shadowkv_integration.py`
- Test against: `sparse_frontier/modelling/attention/shadowkv.py`, `sparse_frontier/modelling/attention/handler.py`, `sparse_frontier/modelling/models/vllm_model.py`

- [x] **Step 1: Write a minimal reference implementation in the test module.**

Use small float32 tensors and the same operations as the official `ShadowKVCache`: flatten all KV heads before `torch.svd`, store `U` and `S*V^T`, mean each post-RoPE chunk, select smallest minimum cosine chunks as outliers, preserve remaining landmark order, softmax landmark scores per GQA group, max across groups, and gather chunk token positions.

- [x] **Step 2: Add failing tests for state equivalence.**

Test that the production class matches reference outlier indices, landmark indices, `U`, local `SV`, and selected chunk indices. Use a prompt length satisfying the official constraints, `chunk_size=2`, `local_chunk=1`, `outlier_chunk=1`, and `sparse_budget=4` so the test stays small while exercising every branch.

- [x] **Step 3: Add failing tests for reconstructed keys and attention output.**

Use a deterministic scalar-position RoPE callable, compare the selected reconstructed keys against reference `U[position] @ SV[head]` followed by the same RoPE, and compare the temporary-cache FlashAttention call's logical output against a float32 reference attention over `[local, outlier, selected, generated]` in that exact order.

- [x] **Step 4: Add failing tests for validation and reset.**

Assert clear `ValueError`s for nonpositive parameters, a budget not divisible by chunk size, a rank larger than the SVD dimension, and a prompt shorter than `(local_chunk + outlier_chunk + sparse_budget // chunk_size) * chunk_size`. Assert `reset()` clears every per-request/per-layer state tensor and restores `initialized=False`.

- [x] **Step 5: Add failing integration tests for the vLLM hook and registry.**

Use a fake `vllm` module only where import isolation is required. Assert the registry contains `shadowkv`, the YAML defaults are exact, the rotary hook captures a cloned pre-RoPE K plus positions before the wrapped RoPE mutates K, scalar positions are accepted, and multidimensional positions produce an error naming unsupported RoPE architectures.

- [x] **Step 6: Run the focused tests and record the expected red failure.**

Run:

```bash
python3 -m pytest -q tests/test_shadowkv_algorithm.py tests/test_shadowkv_integration.py
```

Expected: collection or assertion failures because `ShadowKVAttention`, the registry entry, and the capture hook do not yet exist.

### Task 2: Implement the official ShadowKV state and decode selection

**Files:**
- Create: `sparse_frontier/modelling/attention/shadowkv.py`
- Modify: `sparse_frontier/modelling/attention/registry.py`

- [x] **Step 1: Add constructor validation and explicit state tensors.**

Expose `sparse_budget`, `chunk_size`, `rank`, `local_chunk`, and `outlier_chunk`; require positive integer values and budget divisibility; retain only `[layers, seq, rank]` `U`, `[layers, local_kv, rank, head_dim]` `SV`, local post-RoPE landmarks/indices, outlier indices, and selected chunk indices.

- [x] **Step 2: Implement prefill state construction.**

Gather pre-RoPE local KV heads across tensor parallel workers along the head dimension, flatten to `[1, prompt_len, global_kv_heads * head_dim]`, run `torch.svd(k.float())`, cast/store the official truncated factors, then compute local post-RoPE chunk means and the official minimum-cosine outlier top-k. Preserve the official remaining landmark order and retain prompt RoPE positions for later reconstruction. Do not compress or alter the dense vLLM KV cache.

- [x] **Step 3: Implement exact dense prefill.**

Make `__call__` return `AttentionUtils.flash_attention` over the full post-RoPE prefill tensors. It must not use the low-rank or selected state for prefill outputs.

- [x] **Step 4: Implement official GQA landmark retrieval.**

For a single post-RoPE query, reshape it as `[batch, local_kv_heads, q_per_kv, q_len, head_dim]`, compute landmark dot products divided by `sqrt(head_dim)`, softmax in float32 and cast back, sum over query positions, max across GQA groups when there is more than one, top-k `sparse_budget // chunk_size`, and gather the corresponding original chunk indices.

- [x] **Step 5: Implement key reconstruction and cache-backed value assembly.**

Map selected chunks to token positions, gather local/outlier/selected/generated values and post-RoPE local/outlier/generated keys from the existing vLLM cache, reconstruct selected pre-RoPE keys from `U` and local `SV`, apply the captured original RoPE callable at the saved original positions, and concatenate in official order `[local, outlier, selected, generated]`.

- [x] **Step 6: Invoke existing FlashAttention for decode.**

Assemble the selected, local, outlier, and generated tensors into reusable head-major buffers, transpose into reusable varlen scratch, and call vLLM's `flash_attn_varlen_func` with GQA, causal masking, and the caller's output buffer. This avoids a padded block-cache allocation/copy on every decode layer.

- [x] **Step 7: Run the focused algorithm tests and make the smallest refactors needed.**

Run the two focused test files again; expected result is green for state/index/reconstruction/output equivalence and validation/reset behavior.

### Task 3: Wire pre-RoPE capture into the existing handler and vLLM patch

**Files:**
- Modify: `sparse_frontier/modelling/attention/handler.py`
- Modify: `sparse_frontier/modelling/models/vllm_model.py`
- Modify: `sparse_frontier/modelling/attention/registry.py`

- [x] **Step 1: Add handler capture and consume methods.**

Capture a detached clone of pre-RoPE K, positions, and the bound vLLM RoPE dispatch method; normalize flattened K to `[tokens, local_kv_heads, head_dim]`; consume exactly once per layer; raise a clear ShadowKV-specific error if no capture was produced.

- [x] **Step 2: Reset ShadowKV state at each new prefill without losing layer-0 capture.**

Call the attention object's `reset()` from `_begin_prefill`, clear stale consumed state, and keep the current layer-0 capture until the first handler call consumes it. Continue resetting token counters and reporting state as before.

- [x] **Step 3: Pass captured state through prefill and decode.**

On prefill, call `prefill_state` before dense output computation and then update the unchanged vLLM cache. On decode, append generated K/V first through the existing update function, then call ShadowKV selection/reconstruction with the captured RoPE dispatch method and the reshaped cache.

- [x] **Step 4: Add a ShadowKV-only scalar RoPE hook.**

In `swap_vllm_attention`, patch `vllm.model_executor.layers.rotary_embedding.RotaryEmbedding.__call__` once, only when `SF_ATTENTION_NAME=shadowkv`. Capture before invoking the original callable so in-place RoPE cannot destroy pre-RoPE K. Reject missing K or non-1D positions with an explicit unsupported-RoPE error; leave all non-ShadowKV RoPE/model behavior unchanged.

- [x] **Step 5: Register the constructor with global model/TP metadata.**

Pass model layer/head counts, TP size, and cache block size through the existing registry initialization. Require divisible query/KV head partitioning for this first correctness path and verify all-gather returns the configured global KV-head count.

- [x] **Step 6: Run integration tests and syntax checks.**

Run:

```bash
python3 -m pytest -q tests/test_shadowkv_algorithm.py tests/test_shadowkv_integration.py
python3 -m compileall -q sparse_frontier tests
```

Expected: focused tests pass and `compileall` exits 0.

### Task 4: Add configuration and accuracy-only documentation

**Files:**
- Create: `sparse_frontier/configs/attention/shadowkv.yaml`
- Modify: `README.md`

- [x] **Step 1: Add the exact default configuration.**

Set `sparse_budget: 2048`, `chunk_size: 8`, `rank: 160`, `local_chunk: 4`, and `outlier_chunk: 48` under `attention.args`.

- [x] **Step 2: Document scope and deviations.**

State that this is the official `ShadowKVCache` correctness/accuracy path, prefill is exact/dense, the vLLM cache remains dense, CPU offloading/custom CUDA kernels are intentionally not ported, and no memory savings are claimed. Document the supported scalar RoPE requirement and the clear short-context constraint.

- [x] **Step 3: Test Hydra/config loading without a model run.**

Run a direct YAML parse and the focused integration tests; expected values are the five exact defaults and a registered `shadowkv` implementation.

### Task 5: Run the verification ladder and report evidence

**Files/artifacts:**
- Test logs from focused unit/integration tests
- `compileall` output
- Optional GPU smoke log from Llama-3.1-8B RULER

- [x] **Step 1: Run static/import/config verification.**

Run `python3 -m compileall -q sparse_frontier tests` and the focused pytest command; record exit codes and test counts.

- [x] **Step 2: Run a one-batch numerical verification.**

Use the deterministic reference tests to establish state/index/reconstruction/output equivalence, including a TP all-gather stub that verifies global SVD input and local `SV` slicing.

- [ ] **Step 3: Run the requested end-to-end smoke test when CUDA/vLLM/model access exists.**

Run one Llama-3.1-8B RULER sample with `attention=shadowkv`, a prompt at least 2464 tokens, default ShadowKV args, and the repository's vLLM v1 launcher. Save stdout/stderr and report whether the smoke artifact completed. If dependencies/model/GPU are unavailable, report that exact blocker rather than claiming end-to-end verification.

- [x] **Step 4: Re-read the requirements and report any deviation.**

Explicitly distinguish verified equivalence from unavailable end-to-end execution and note that dense vLLM cache storage means this baseline does not claim memory savings.
