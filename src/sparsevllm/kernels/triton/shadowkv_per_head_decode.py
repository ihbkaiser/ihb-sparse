"""Fixed-grid decode for ShadowKV payloads with per-KV-head positions.

The ordinary paged decode contract has one token table per request. ShadowKV
selection is query-aware per KV head, so its materialized payload is instead
laid out as ``[batch, kv_heads, compact_tokens, head_dim]`` with one effective
length per ``(batch, kv_head)``. The kernels below keep the same split-KV
reduction as the regular fixed-grid Triton provider while reading that
head-indexed length and payload directly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _shadowkv_per_head_stage1(
    Q,
    K,
    V,
    HeadLens,
    sm_scale,
    Mid_O,
    Mid_Lse,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_hlb,
    stride_hlh,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    GQA_GROUP_SIZE: tl.constexpr,
    QUERY_HEAD_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_EFFECTIVE_SPLITS: tl.constexpr,
    TARGET_TOKENS_PER_SPLIT: tl.constexpr,
):
    batch_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    split_id = tl.program_id(2)

    head_offsets = tl.arange(0, QUERY_HEAD_BLOCK)
    query_heads = kv_head_id * GQA_GROUP_SIZE + head_offsets
    head_mask = head_offsets < GQA_GROUP_SIZE
    seq_len = tl.load(
        HeadLens + batch_id * stride_hlb + kv_head_id * stride_hlh
    )
    requested_splits = tl.cdiv(seq_len, TARGET_TOKENS_PER_SPLIT)
    num_splits = tl.maximum(
        1, tl.minimum(requested_splits, MAX_EFFECTIVE_SPLITS)
    )
    split_tokens = tl.where(
        requested_splits <= MAX_EFFECTIVE_SPLITS,
        TARGET_TOKENS_PER_SPLIT,
        tl.cdiv(seq_len, num_splits),
    )
    split_start = split_id * split_tokens
    split_end = tl.minimum(split_start + split_tokens, seq_len)
    split_valid = (split_id < num_splits) & (split_start < split_end)
    if not split_valid:
        return

    offs_d = tl.arange(0, HEAD_DIM)
    q_offsets = (
        batch_id * stride_qb
        + query_heads[:, None] * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + q_offsets, mask=head_mask[:, None], other=0.0)
    max_logit = tl.zeros([QUERY_HEAD_BLOCK], dtype=tl.float32) - float("inf")
    exp_sum = tl.zeros([QUERY_HEAD_BLOCK], dtype=tl.float32)
    acc = tl.zeros([QUERY_HEAD_BLOCK, HEAD_DIM], dtype=tl.float32)
    block_count = tl.where(
        split_valid, tl.cdiv(split_end - split_start, BLOCK_N), 0
    )

    for block_id in range(0, block_count):
        positions = split_start + block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        position_mask = positions < split_end
        k_offsets = (
            batch_id * stride_kb
            + kv_head_id * stride_kh
            + positions[None, :] * stride_ks
            + offs_d[:, None]
        )
        v_offsets = (
            batch_id * stride_vb
            + kv_head_id * stride_vh
            + positions[:, None] * stride_vs
            + offs_d[None, :]
        )
        k = tl.load(K + k_offsets, mask=position_mask[None, :], other=0.0)
        v = tl.load(V + v_offsets, mask=position_mask[:, None], other=0.0)
        logits = tl.dot(q, k)
        logits = tl.where(position_mask[None, :], logits, -float("inf"))
        logits *= sm_scale

        block_max = tl.max(logits, axis=1)
        next_max = tl.maximum(max_logit, block_max)
        old_scale = tl.exp(max_logit - next_max)
        probs = tl.exp(logits - next_max[:, None])
        acc = acc * old_scale[:, None] + tl.dot(probs.to(v.dtype), v)
        exp_sum = exp_sum * old_scale + tl.sum(probs, axis=1)
        max_logit = next_max

    safe_sum = tl.where(split_valid, exp_sum, 1.0)
    mid_offsets = (
        batch_id * stride_mid_b
        + query_heads[:, None] * stride_mid_h
        + split_id * stride_mid_s
        + offs_d[None, :]
    )
    lse_offsets = (
        batch_id * stride_lse_b
        + query_heads * stride_lse_h
        + split_id * stride_lse_s
    )
    tl.store(
        Mid_O + mid_offsets,
        tl.where(split_valid, acc / safe_sum[:, None], 0.0),
        mask=head_mask[:, None],
    )
    tl.store(
        Mid_Lse + lse_offsets,
        tl.where(split_valid, max_logit + tl.log(safe_sum), -float("inf")),
        mask=head_mask,
    )


@triton.jit
def _shadowkv_per_head_stage2(
    HeadLens,
    Mid_O,
    Mid_Lse,
    O,
    Out_Lse,
    stride_hlb,
    stride_hlh,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    stride_ob,
    stride_oh,
    stride_out_lse_h,
    stride_out_lse_b,
    HEAD_DIM: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    MAX_EFFECTIVE_SPLITS: tl.constexpr,
    TARGET_TOKENS_PER_SPLIT: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    kv_head_id = head_id // GQA_GROUP_SIZE
    seq_len = tl.load(
        HeadLens + batch_id * stride_hlb + kv_head_id * stride_hlh
    )
    num_splits = tl.maximum(
        1,
        tl.minimum(
            tl.cdiv(seq_len, TARGET_TOKENS_PER_SPLIT),
            MAX_EFFECTIVE_SPLITS,
        ),
    )

    offs_d = tl.arange(0, HEAD_DIM)
    mid_base = batch_id * stride_mid_b + head_id * stride_mid_h + offs_d
    lse_base = batch_id * stride_lse_b + head_id * stride_lse_h
    max_lse = -float("inf")
    exp_sum = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for split_id in range(0, num_splits):
        split_lse = tl.load(Mid_Lse + lse_base + split_id * stride_lse_s)
        split_o = tl.load(Mid_O + mid_base + split_id * stride_mid_s)
        next_max = tl.maximum(max_lse, split_lse)
        old_scale = tl.exp(max_lse - next_max)
        split_scale = tl.exp(split_lse - next_max)
        acc = acc * old_scale + split_scale * split_o
        exp_sum = exp_sum * old_scale + split_scale
        max_lse = next_max

    tl.store(
        O + batch_id * stride_ob + head_id * stride_oh + offs_d,
        acc / exp_sum,
    )
    tl.store(
        Out_Lse + head_id * stride_out_lse_h + batch_id * stride_out_lse_b,
        max_lse + tl.log(exp_sum),
    )


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_lens: torch.Tensor,
    mid_o: torch.Tensor,
    mid_lse: torch.Tensor,
) -> None:
    if q.ndim != 3 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("ShadowKV per-head decode expects Q[batch,heads,dim] and K/V[batch,kv_heads,tokens,dim].")
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise TypeError("ShadowKV per-head query, key, and value dtypes must match.")
    if tuple(k.shape) != tuple(v.shape):
        raise ValueError("ShadowKV per-head K/V shapes must match.")
    if int(q.shape[0]) != int(k.shape[0]) or int(q.shape[1]) % int(k.shape[1]):
        raise ValueError("ShadowKV per-head batch and GQA dimensions disagree.")
    if head_lens.dtype != torch.int32 or head_lens.ndim != 2:
        raise TypeError("ShadowKV per-head lengths must be a contiguous int32 [batch,kv_heads] tensor.")
    if tuple(head_lens.shape) != tuple(k.shape[:2]) or not head_lens.is_contiguous():
        raise ValueError("ShadowKV per-head lengths must match the K/V batch and KV-head axes.")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("ShadowKV per-head head dimensions must be contiguous.")
    if mid_o.ndim != 4 or mid_lse.ndim != 3 or tuple(mid_o.shape[:3]) != tuple(mid_lse.shape):
        raise ValueError("ShadowKV per-head split workspaces have incompatible shapes.")
    if tuple(mid_o.shape[:2]) != tuple(q.shape[:2]):
        raise ValueError("ShadowKV per-head split workspace head axes disagree with Q.")


@torch.no_grad()
def shadowkv_per_head_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_lens: torch.Tensor,
    mid_o: torch.Tensor,
    mid_lse: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    target_tokens_per_split: int,
    block_n: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
    stage2_num_warps: int = 4,
    stage2_num_stages: int = 2,
    output: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    return_softmax_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run fixed-grid split-KV decode over per-KV-head compact payloads."""

    _check_inputs(q, k, v, head_lens, mid_o, mid_lse)
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("ShadowKV per-head decode requires CUDA tensors.")
    head_dim = int(q.shape[-1])
    if head_dim not in {16, 32, 64, 128, 256}:
        raise ValueError(f"unsupported ShadowKV per-head head_dim={head_dim}")
    if int(block_n) not in {16, 32, 64, 128}:
        raise ValueError(f"unsupported ShadowKV per-head BLOCK_N={block_n}")
    if int(target_tokens_per_split) <= 0 or int(num_warps) <= 0 or int(num_stages) <= 0:
        raise ValueError("ShadowKV per-head launch parameters must be positive.")
    if int(stage2_num_warps) <= 0 or int(stage2_num_stages) <= 0:
        raise ValueError("ShadowKV per-head stage2 launch parameters must be positive.")
    batch, num_heads, _ = map(int, q.shape)
    kv_heads = int(k.shape[1])
    group_size = num_heads // kv_heads
    max_splits = int(mid_o.shape[2])
    if max_splits <= 0:
        raise ValueError("ShadowKV per-head split workspace must be non-empty.")
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    if softmax_scale <= 0:
        raise ValueError("ShadowKV per-head softmax_scale must be positive.")
    if output is None:
        output = torch.empty_like(q)
    elif tuple(output.shape) != tuple(q.shape) or output.dtype != q.dtype or output.device != q.device:
        raise ValueError("ShadowKV per-head output must match Q shape, dtype, and device.")
    if output_lse is None:
        output_lse = torch.empty((num_heads, batch), dtype=torch.float32, device=q.device)
    elif tuple(output_lse.shape) != (num_heads, batch) or output_lse.dtype != torch.float32:
        raise ValueError("ShadowKV per-head output_lse must be [query_heads,batch] FP32.")

    _shadowkv_per_head_stage1[(batch, kv_heads, max_splits)](
        q,
        k,
        v,
        head_lens,
        float(softmax_scale),
        mid_o,
        mid_lse,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        head_lens.stride(0),
        head_lens.stride(1),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        GQA_GROUP_SIZE=group_size,
        QUERY_HEAD_BLOCK=max(16, triton.next_power_of_2(group_size)),
        HEAD_DIM=head_dim,
        BLOCK_N=int(block_n),
        MAX_EFFECTIVE_SPLITS=max_splits,
        TARGET_TOKENS_PER_SPLIT=int(target_tokens_per_split),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _shadowkv_per_head_stage2[(batch, num_heads)](
        head_lens,
        mid_o,
        mid_lse,
        output,
        output_lse,
        head_lens.stride(0),
        head_lens.stride(1),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        output.stride(0),
        output.stride(1),
        output_lse.stride(0),
        output_lse.stride(1),
        HEAD_DIM=head_dim,
        GQA_GROUP_SIZE=group_size,
        MAX_EFFECTIVE_SPLITS=max_splits,
        TARGET_TOKENS_PER_SPLIT=int(target_tokens_per_split),
        num_warps=int(stage2_num_warps),
        num_stages=int(stage2_num_stages),
    )
    return (output, output_lse) if return_softmax_lse else output


__all__ = ["shadowkv_per_head_decode"]
