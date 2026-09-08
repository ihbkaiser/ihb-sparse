import os

import pytest
import torch

from sparsevllm.kernels.shadowkv_cutlass import load_shadowkv_cutlass


_CUTLASS_ROOT = os.environ.get("SPARSEVLLM_CUTLASS_ROOT")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _CUTLASS_ROOT,
    reason="requires CUDA and SPARSEVLLM_CUTLASS_ROOT",
)
def test_shadowkv_cutlass_score_and_reconstruction_match_oracles():
    kernel = load_shadowkv_cutlass(_CUTLASS_ROOT)
    torch.manual_seed(20260907)
    batch, heads, groups, candidates, dim = 2, 2, 4, 40, 128
    query = torch.randn(
        batch, heads, groups, dim, device="cuda", dtype=torch.bfloat16
    )
    landmarks = torch.randn(
        batch, heads, candidates, dim, device="cuda", dtype=torch.bfloat16
    )
    scores = torch.empty(
        batch, heads, groups, candidates, device="cuda", dtype=torch.bfloat16
    )
    kernel.batch_gemm_softmax(query, landmarks, scores, dim**-0.5)
    reference = torch.softmax(
        torch.einsum("bhgd,bhnd->bhgn", query.float(), landmarks.float())
        * dim**-0.5,
        dim=-1,
    )
    torch.testing.assert_close(scores.float(), reference, rtol=3e-2, atol=3e-2)

    # Exercise the vendored ShadowKV BatchGemmSoftmax operator as well as the
    # small compatibility entry point above.  The former keeps the GEMM,
    # partial max/sum reduction, and final softmax in one CUTLASS pipeline.
    logits = torch.empty_like(scores)
    fused_scores = torch.empty_like(scores)
    norm = torch.empty(batch, heads, groups, device="cuda", dtype=torch.float32)
    total = torch.empty_like(norm)
    kernel.batch_gemm_softmax_exact(
        query, landmarks, logits, fused_scores, norm, total, dim**-0.5
    )
    torch.testing.assert_close(
        fused_scores.float(), reference, rtol=3e-2, atol=3e-2
    )

    # The exact selector is part of ShadowKV's decode graph.  Its CUTLASS
    # opt-in shared-memory attribute must therefore be configured before
    # capture, not from the run path.
    graph_query = query.clone()
    graph_landmarks = landmarks.clone()
    graph_logits = torch.empty_like(logits)
    graph_probabilities = torch.empty_like(fused_scores)
    graph_norm = torch.empty_like(norm)
    graph_sum = torch.empty_like(total)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        kernel.batch_gemm_softmax_exact(
            graph_query,
            graph_landmarks,
            graph_logits,
            graph_probabilities,
            graph_norm,
            graph_sum,
            dim**-0.5,
        )
    graph_query.copy_(query.roll(1, dims=2))
    graph_landmarks.copy_(landmarks.roll(2, dims=2))
    graph.replay()
    torch.cuda.synchronize()
    graph_reference = torch.softmax(
        torch.einsum(
            "bhgd,bhnd->bhgn",
            graph_query.float(),
            graph_landmarks.float(),
        )
        * dim**-0.5,
        dim=-1,
    )
    torch.testing.assert_close(
        graph_probabilities.float(), graph_reference, rtol=3e-2, atol=3e-2
    )

    width, rank = 11, 160
    sources = [
        torch.randn(32, 1, rank, dtype=torch.bfloat16, pin_memory=True)
        for _ in range(batch)
    ]
    positions = torch.tensor(
        [[0, 3, -1, 7, 8, 9, 10, 11, 12, 13, 14],
         [4, 5, 17, 31, -1, 2, 3, 6, 7, 8, 9]],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor([32, 20], dtype=torch.int32, device="cuda")
    sv = torch.randn(batch, rank, heads, dim, device="cuda", dtype=torch.bfloat16)
    cos_sin = torch.zeros(32, dim, device="cuda", dtype=torch.float32)
    cos_sin[:, : dim // 2] = 1
    pointers = torch.empty(batch, dtype=torch.int64, device="cuda")
    u = torch.empty(batch, width, 1, rank, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(
        batch, width, heads, dim, dtype=torch.bfloat16, device="cuda"
    )
    kernel.set_host_pointers(sources, pointers)
    kernel.batch_gather_gemm(pointers, positions, lengths, sv, cos_sin, u, output)
    expected_u = torch.zeros_like(u)
    for batch_idx in range(batch):
        for token_idx, position in enumerate(positions[batch_idx].cpu().tolist()):
            if 0 <= position < int(lengths[batch_idx]):
                expected_u[batch_idx, token_idx, 0].copy_(sources[batch_idx][position, 0])
    expected = torch.bmm(
        expected_u[:, :, 0].float(), sv.float().reshape(batch, rank, heads * dim)
    ).view_as(output)
    torch.testing.assert_close(u, expected_u)
    torch.testing.assert_close(output.float(), expected.float(), rtol=4e-2, atol=4e-2)

    head_positions = positions[:, None, :]
    head_lengths = lengths[:, None]
    head_output = torch.empty(
        batch, 1, width, rank, dtype=torch.bfloat16, device="cuda"
    )
    kernel.gather_host_per_head(pointers, head_positions, head_lengths, head_output)
    expected_head = torch.zeros_like(head_output)
    for batch_idx in range(batch):
        for token_idx, position in enumerate(head_positions[batch_idx, 0].cpu().tolist()):
            if 0 <= position < int(head_lengths[batch_idx, 0].item()):
                expected_head[batch_idx, 0, token_idx].copy_(sources[batch_idx][position, 0])
    torch.testing.assert_close(head_output, expected_head)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _CUTLASS_ROOT,
    reason="requires CUDA and SPARSEVLLM_CUTLASS_ROOT",
)
def test_shadowkv_cutlass_fused_gather_gemm_matches_oracle_and_graph_replay():
    kernel = load_shadowkv_cutlass(_CUTLASS_ROOT)
    torch.manual_seed(20260908)
    batch, heads, prompt_capacity, chunk_size = 2, 2, 32, 8
    chunks, width, rank, dim = 16, 128, 160, 128
    u_device = torch.randn(
        batch, prompt_capacity, rank, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    sv_column_major = torch.randn(
        batch, heads, dim, rank, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    chunk_indices = torch.randint(
        0, prompt_capacity // chunk_size, (batch, heads, chunks),
        device="cuda", dtype=torch.int32,
    ).contiguous()
    positions = torch.zeros(
        batch, heads, width, device="cuda", dtype=torch.int32
    )
    lengths = torch.full(
        (batch, heads), prompt_capacity, device="cuda", dtype=torch.int32
    )
    cos_sin = torch.zeros(prompt_capacity, dim, device="cuda", dtype=torch.float32)
    cos_sin[:, : dim // 2] = 1
    output = torch.empty(
        batch * heads, width, 1, dim, device="cuda", dtype=torch.bfloat16
    )

    kernel.batch_gather_gemm_fused_rope(
        u_device, sv_column_major, chunk_indices, positions, lengths,
        cos_sin, output, prompt_capacity, chunk_size,
    )
    token_indices = (
        chunk_indices.long()[..., None] * chunk_size
        + torch.arange(chunk_size, device="cuda")[None, None, None, :]
    ).reshape(batch, heads, width)
    selected_u = u_device.float()[:, None].expand(
        batch, heads, prompt_capacity, rank
    ).gather(
        2, token_indices[..., None].expand(batch, heads, width, rank)
    )
    expected = torch.einsum(
        "bhwr,bhrd->bhwd", selected_u, sv_column_major.float().transpose(-1, -2)
    ).reshape(batch * heads, width, 1, dim)
    torch.testing.assert_close(output.float(), expected, rtol=4e-2, atol=4e-2)

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        graph.capture_begin()
        kernel.batch_gather_gemm_fused_rope(
            u_device, sv_column_major, chunk_indices, positions, lengths,
            cos_sin, output, prompt_capacity, chunk_size,
        )
        graph.capture_end()
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(output.float(), expected, rtol=4e-2, atol=4e-2)
