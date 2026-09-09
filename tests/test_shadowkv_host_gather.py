from types import SimpleNamespace

import pytest
import torch

from sparsevllm.engine.cache_manager.shadowkv import ShadowKVCacheManager
from sparsevllm.utils.context import reset_context, set_context
from sparsevllm.kernels.shadowkv_host_gather import load_shadowkv_host_gather


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shadowkv_host_gather_matches_cpu_oracle_and_cuda_graph_replay():
    torch.manual_seed(20260906)
    batch, width, heads, head_dim = 2, 9, 8, 128
    sources = [
        torch.randn(32, heads, head_dim, dtype=torch.bfloat16, pin_memory=True)
        for _ in range(batch)
    ]
    positions = torch.tensor(
        [[0, 7, -1, 12, 31, 2, 3, 4, 5], [4, 8, 16, -1, 20, 21, 22, 23, 24]],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor([32, 25], dtype=torch.int32, device="cuda")
    output = torch.empty(
        batch, width, heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    pointers = torch.empty(batch, dtype=torch.int64, device="cuda")
    kernel = load_shadowkv_host_gather()
    kernel.set_host_pointers(sources, pointers)
    kernel.gather_host(pointers, positions, lengths, output)
    torch.cuda.synchronize()

    expected = torch.zeros_like(output)
    for batch_idx in range(batch):
        for token_idx, position in enumerate(positions[batch_idx].cpu().tolist()):
            if 0 <= position < int(lengths[batch_idx].item()):
                expected[batch_idx, token_idx].copy_(sources[batch_idx][position])
    torch.testing.assert_close(output, expected)

    graph_positions = positions.clone()
    graph_output = torch.empty_like(output)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        kernel.gather_host(pointers, graph_positions, lengths, graph_output)
    graph_positions.copy_(
        torch.tensor(
            [[31, 30, 29, 28, 27, 26, -1, 1, 0], [24, 23, 22, 21, 20, 19, 18, 17, -1]],
            dtype=torch.int32,
            device="cuda",
        )
    )
    graph.replay()
    torch.cuda.synchronize()

    expected.zero_()
    for batch_idx in range(batch):
        for token_idx, position in enumerate(graph_positions[batch_idx].cpu().tolist()):
            if 0 <= position < int(lengths[batch_idx].item()):
                expected[batch_idx, token_idx].copy_(sources[batch_idx][position])
    torch.testing.assert_close(graph_output, expected)

    # ShadowKV's low-rank U can have a non-vector-aligned rank. The scalar
    # tail path must preserve the same host-gather contract.
    u_sources = [
        torch.arange(20, dtype=torch.bfloat16, pin_memory=True).reshape(5, 1, 4),
        torch.arange(24, dtype=torch.bfloat16, pin_memory=True).reshape(6, 1, 4),
    ]
    u_positions = torch.tensor(
        [[0, 3, -1], [5, 2, 7]], dtype=torch.int32, device="cuda"
    )
    u_lengths = torch.tensor([5, 6], dtype=torch.int32, device="cuda")
    u_output = torch.empty(2, 3, 1, 4, dtype=torch.bfloat16, device="cuda")
    u_pointers = torch.empty(2, dtype=torch.int64, device="cuda")
    kernel.set_host_pointers(u_sources, u_pointers)
    kernel.gather_host(u_pointers, u_positions, u_lengths, u_output)
    torch.cuda.synchronize()
    u_expected = torch.zeros_like(u_output)
    for batch_idx in range(2):
        for token_idx, position in enumerate(u_positions[batch_idx].cpu().tolist()):
            if 0 <= position < int(u_lengths[batch_idx].item()):
                u_expected[batch_idx, token_idx].copy_(u_sources[batch_idx][position])
    torch.testing.assert_close(u_output, u_expected)

    rank = 4
    sv_bmm = torch.randn(
        2, rank, 2, 4, dtype=torch.bfloat16, device="cuda"
    )
    cos_sin = torch.randn(32, 4, dtype=torch.float32, device="cuda")
    fused_output = torch.empty(2, 3, 2, 4, dtype=torch.bfloat16, device="cuda")
    kernel.gather_gemm_rope(
        u_pointers,
        u_positions,
        u_lengths,
        sv_bmm,
        cos_sin,
        u_output,
        fused_output,
    )
    torch.cuda.synchronize()

    raw = torch.bmm(
        u_expected[:, :, 0, :].float(),
        sv_bmm.float().permute(0, 1, 2, 3).reshape(2, rank, 8),
    ).view(2, 3, 2, 4)
    fused_expected = torch.zeros_like(raw)
    for batch_idx in range(2):
        for token_idx, position in enumerate(u_positions[batch_idx].cpu().tolist()):
            if 0 <= position < int(u_lengths[batch_idx].item()):
                rope = cos_sin[position]
                cos, sin = rope.chunk(2)
                x1 = raw[batch_idx, token_idx, :, :2]
                x2 = raw[batch_idx, token_idx, :, 2:]
                fused_expected[batch_idx, token_idx, :, :2] = x1 * cos[:2] - x2 * sin[:2]
                fused_expected[batch_idx, token_idx, :, 2:] = x2 * cos[:2] + x1 * sin[:2]
    torch.testing.assert_close(
        fused_output.float(), fused_expected.float(), rtol=3e-2, atol=4e-2
    )

    # The fused entry point must keep its BMM and RoPE launch graph-safe.
    graph_positions = u_positions.clone()
    graph_fused_output = torch.empty_like(fused_output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel.gather_gemm_rope(
            u_pointers,
            graph_positions,
            u_lengths,
            sv_bmm,
            cos_sin,
            u_output,
            graph_fused_output,
        )
    graph_positions.copy_(
        torch.tensor([[4, 1, -1], [3, 0, 2]], dtype=torch.int32, device="cuda")
    )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(graph_fused_output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shadowkv_graph_recent_kv_write_is_replay_safe():
    manager = object.__new__(ShadowKVCacheManager)
    manager.config = SimpleNamespace(decode_graph=True)
    manager.num_kv_heads = 2
    manager.head_dim = 4
    manager.device = torch.device("cuda")
    manager._shadow_decode_workspaces = {
        0: {
            "recent_k": torch.zeros(2, 4, 2, 4, dtype=torch.float16, device="cuda"),
            "recent_v": torch.zeros(2, 4, 2, 4, dtype=torch.float16, device="cuda"),
            "prompt_lens": torch.tensor([3, 4], dtype=torch.int32, device="cuda"),
        }
    }
    manager.layer_batch_state = SimpleNamespace(
        context_lens=torch.tensor([4, 6], dtype=torch.int32, device="cuda")
    )

    try:
        set_context(is_prefill=False)
        k = torch.arange(16, device="cuda", dtype=torch.float16).reshape(2, 2, 4)
        v = -k
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            manager.save_rope_kv_if_needed(0, k, v)

        k_next = k + 100
        v_next = v - 100
        manager.layer_batch_state.context_lens.copy_(
            torch.tensor([5, 7], dtype=torch.int32, device="cuda")
        )
        k.copy_(k_next)
        v.copy_(v_next)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(manager._shadow_decode_workspaces[0]["recent_k"][0, 1], k_next[0])
        torch.testing.assert_close(manager._shadow_decode_workspaces[0]["recent_k"][1, 2], k_next[1])
        torch.testing.assert_close(manager._shadow_decode_workspaces[0]["recent_v"][0, 1], v_next[0])
        torch.testing.assert_close(manager._shadow_decode_workspaces[0]["recent_v"][1, 2], v_next[1])
    finally:
        reset_context()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shadowkv_per_head_host_gather_matches_cpu_oracle_and_graph_replay():
    torch.manual_seed(20260907)
    batch, heads, width, head_dim = 2, 3, 7, 8
    sources = [
        torch.randn(19, heads, head_dim, dtype=torch.bfloat16, pin_memory=True)
        for _ in range(batch)
    ]
    positions = torch.tensor(
        [
            [[0, 3, -1, 7, 12, 1, 2], [2, 4, 6, -1, 8, 9, 10], [1, 5, 11, 13, -1, 14, 15]],
            [[3, 5, 7, 9, -1, 2, 4], [0, 1, 2, 3, 4, -1, 6], [8, 10, 12, -1, 14, 16, 18]],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor([[13, 11, 16], [10, 7, 19]], dtype=torch.int32, device="cuda")
    output = torch.empty(batch, heads, width, head_dim, dtype=torch.bfloat16, device="cuda")
    pointers = torch.empty(batch, dtype=torch.int64, device="cuda")
    kernel = load_shadowkv_host_gather()
    kernel.set_host_pointers(sources, pointers)
    kernel.gather_host_per_head(pointers, positions, lengths, output)
    torch.cuda.synchronize()

    expected = torch.zeros_like(output)
    for batch_idx in range(batch):
        for head_idx in range(heads):
            for token_idx, position in enumerate(positions[batch_idx, head_idx].cpu().tolist()):
                if 0 <= position < int(lengths[batch_idx, head_idx].item()):
                    expected[batch_idx, head_idx, token_idx].copy_(
                        sources[batch_idx][position, head_idx]
                    )
    torch.testing.assert_close(output, expected)

    graph_positions = positions.clone()
    graph_output = torch.empty_like(output)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        kernel.gather_host_per_head(pointers, graph_positions, lengths, graph_output)
    graph_positions.copy_(positions.flip(-1))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(graph_output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shadowkv_gpu_cache_per_head_gather_matches_oracle_and_graph_replay():
    """Protect direct GPU-cache lookup and its graph-stable tensor contract."""
    torch.manual_seed(20260907)
    batch, source_width, heads, width, head_dim = 2, 19, 3, 7, 8
    source = torch.randn(
        batch, source_width, heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    positions = torch.tensor(
        [
            [[0, 3, -1, 7, 12, 1, 2], [2, 4, 6, -1, 8, 9, 10], [1, 5, 11, 13, -1, 14, 15]],
            [[3, 5, 7, 9, -1, 2, 4], [0, 1, 2, 3, 4, -1, 6], [8, 10, 12, -1, 14, 16, 18]],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor([13, 7], dtype=torch.int32, device="cuda")
    output = torch.empty(batch, heads, width, head_dim, dtype=source.dtype, device="cuda")
    kernel = load_shadowkv_host_gather()
    kernel.gather_gpu_cache_per_head(source, positions, lengths, output)
    torch.cuda.synchronize()

    expected = torch.zeros_like(output)
    for batch_idx in range(batch):
        for head_idx in range(heads):
            for token_idx, position in enumerate(positions[batch_idx, head_idx].cpu().tolist()):
                if 0 <= position < int(lengths[batch_idx].item()):
                    expected[batch_idx, head_idx, token_idx].copy_(
                        source[batch_idx, position, head_idx]
                    )
    torch.testing.assert_close(output, expected)

    graph_positions = positions.clone()
    graph_output = torch.empty_like(output)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        kernel.gather_gpu_cache_per_head(source, graph_positions, lengths, graph_output)
    graph_positions.copy_(positions.flip(-1))
    graph.replay()
    torch.cuda.synchronize()
    expected.zero_()
    for batch_idx in range(batch):
        for head_idx in range(heads):
            for token_idx, position in enumerate(graph_positions[batch_idx, head_idx].cpu().tolist()):
                if 0 <= position < int(lengths[batch_idx].item()):
                    expected[batch_idx, head_idx, token_idx].copy_(
                        source[batch_idx, position, head_idx]
                    )
    torch.testing.assert_close(graph_output, expected)

    fused_k = torch.empty_like(output)
    fused_v = torch.empty_like(output)
    kernel.gather_gpu_cache_per_head_kv(
        source, source, positions, lengths, fused_k, fused_v
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(fused_k, output)
    torch.testing.assert_close(fused_v, output)

    graph_positions = positions.clone()
    graph_fused_k = torch.empty_like(output)
    graph_fused_v = torch.empty_like(output)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        kernel.gather_gpu_cache_per_head_kv(
            source, source, graph_positions, lengths, graph_fused_k, graph_fused_v
        )
    graph_positions.copy_(positions.flip(-1))
    graph.replay()
    torch.cuda.synchronize()
    expected.zero_()
    for batch_idx in range(batch):
        for head_idx in range(heads):
            for token_idx, position in enumerate(graph_positions[batch_idx, head_idx].cpu().tolist()):
                if 0 <= position < int(lengths[batch_idx].item()):
                    expected[batch_idx, head_idx, token_idx].copy_(
                        source[batch_idx, position, head_idx]
                    )
    torch.testing.assert_close(graph_fused_k, expected)
    torch.testing.assert_close(graph_fused_v, expected)

    source_v = source + 17
    source_k_ptrs = torch.tensor(
        [int(source[idx].data_ptr()) for idx in range(batch)],
        dtype=torch.int64,
        device="cuda",
    )
    source_v_ptrs = torch.tensor(
        [int(source_v[idx].data_ptr()) for idx in range(batch)],
        dtype=torch.int64,
        device="cuda",
    )
    pointer_fused_k = torch.empty_like(output)
    pointer_fused_v = torch.empty_like(output)
    kernel.gather_gpu_cache_per_head_kv_ptrs(
        source_k_ptrs,
        source_v_ptrs,
        positions,
        lengths,
        pointer_fused_k,
        pointer_fused_v,
        source_width,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(pointer_fused_k, output)
    expected_v = torch.zeros_like(output)
    for batch_idx in range(batch):
        for head_idx in range(heads):
            for token_idx, position in enumerate(positions[batch_idx, head_idx].cpu().tolist()):
                if 0 <= position < int(lengths[batch_idx].item()):
                    expected_v[batch_idx, head_idx, token_idx].copy_(
                        source_v[batch_idx, position, head_idx]
                    )
    torch.testing.assert_close(pointer_fused_v, expected_v)

    graph_positions = positions.clone()
    graph_pointer_fused_k = torch.empty_like(output)
    graph_pointer_fused_v = torch.empty_like(output)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        kernel.gather_gpu_cache_per_head_kv_ptrs(
            source_k_ptrs,
            source_v_ptrs,
            graph_positions,
            lengths,
            graph_pointer_fused_k,
            graph_pointer_fused_v,
            source_width,
        )
    graph_positions.copy_(positions.flip(-1))
    graph.replay()
    torch.cuda.synchronize()
    expected.zero_()
    expected_v.zero_()
    for batch_idx in range(batch):
        for head_idx in range(heads):
            for token_idx, position in enumerate(graph_positions[batch_idx, head_idx].cpu().tolist()):
                if 0 <= position < int(lengths[batch_idx].item()):
                    expected[batch_idx, head_idx, token_idx].copy_(
                        source[batch_idx, position, head_idx]
                    )
                    expected_v[batch_idx, head_idx, token_idx].copy_(
                        source_v[batch_idx, position, head_idx]
                    )
    torch.testing.assert_close(graph_pointer_fused_k, expected)
    torch.testing.assert_close(graph_pointer_fused_v, expected_v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shadowkv_offset_copy_reuses_device_chunks_and_handles_host_misses():
    """Protect the ShadowKV-origin hit/miss compaction contract.

    The independent oracle follows the reordered chunk IDs, so this catches
    both stale device-cache hits and incorrect host-pointer misses without
    freezing the kernel's intentionally unordered atomic compaction.
    """
    torch.manual_seed(20260907)
    batch, heads, source_chunks, chunk_size, head_dim = 2, 2, 8, 2, 8
    map_size = 128
    chunk_dim = chunk_size * head_dim
    sources = [
        torch.randn(source_chunks, 1, chunk_dim, dtype=torch.bfloat16, pin_memory=True)
        for _ in range(batch * heads)
    ]
    pointers = torch.empty(batch * heads, dtype=torch.int64, device="cuda")
    current = torch.full(
        (batch, heads, map_size), -1, dtype=torch.int64, device="cuda"
    )
    current[:, :, :source_chunks] = torch.arange(
        source_chunks, dtype=torch.int64, device="cuda"
    )
    cached = torch.full_like(current, -1)
    reordered = torch.full_like(current, -1)
    offsets = torch.empty(batch * heads * map_size, dtype=torch.int32, device="cuda")
    counts = torch.empty(batch * heads, dtype=torch.int32, device="cuda")
    signals = torch.zeros(batch * heads, dtype=torch.int32, device="cuda")
    values = torch.zeros(
        batch * heads, map_size, chunk_dim, dtype=torch.bfloat16, device="cuda"
    )
    temp = torch.empty_like(values)
    kernel = load_shadowkv_host_gather()
    kernel.set_host_pointers(sources, pointers)

    def run_once():
        kernel.reorder_shadowkv_chunk_offsets(
            cached, current, reordered, offsets, counts, batch, heads, map_size
        )
        kernel.gather_copy_with_offsets(
            pointers,
            values,
            temp,
            offsets,
            counts,
            signals,
            batch,
            heads,
            source_chunks,
            map_size,
            chunk_size,
            head_dim,
            map_size,
        )
        cached.copy_(reordered)
        torch.cuda.synchronize()

    run_once()
    for block in range(batch * heads):
        expected = torch.zeros_like(values[block])
        for item, chunk_id in enumerate(reordered.view(-1, map_size)[block].cpu().tolist()):
            if 0 <= chunk_id < source_chunks:
                expected[item].copy_(sources[block][chunk_id, 0])
        torch.testing.assert_close(values[block], expected)

    # The same two-stage update is part of the CUDA-graph decode path.  The
    # graph must keep the per-layer device cache and metadata addresses fixed
    # while accepting a new selection on replay.
    graph_current = current.clone()
    graph_cached = torch.full_like(cached, -1)
    graph_reordered = torch.full_like(graph_current, -1)
    graph_offsets = torch.empty_like(offsets)
    graph_counts = torch.empty_like(counts)
    graph_signals = torch.zeros_like(signals)
    graph_values = torch.zeros_like(values)
    graph_temp = torch.empty_like(temp)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        kernel.reorder_shadowkv_chunk_offsets(
            graph_cached,
            graph_current,
            graph_reordered,
            graph_offsets,
            graph_counts,
            batch,
            heads,
            map_size,
        )
        kernel.gather_copy_with_offsets(
            pointers,
            graph_values,
            graph_temp,
            graph_offsets,
            graph_counts,
            graph_signals,
            batch,
            heads,
            source_chunks,
            map_size,
            chunk_size,
            head_dim,
            map_size,
        )
        graph_cached.copy_(graph_reordered)
    graph_current.fill_(-1)
    graph_current[:, :, 20 : 20 + source_chunks] = torch.arange(
        source_chunks, dtype=torch.int64, device="cuda"
    )
    graph.replay()
    torch.cuda.synchronize()
    for block in range(batch * heads):
        expected = torch.zeros_like(graph_values[block])
        for item, chunk_id in enumerate(
            graph_reordered.view(-1, map_size)[block].cpu().tolist()
        ):
            if 0 <= chunk_id < source_chunks:
                expected[item].copy_(sources[block][chunk_id, 0])
        torch.testing.assert_close(graph_values[block], expected)

    # Deliberately make the cached slots a reverse permutation.  Every valid
    # item below is then a device-to-device hit with an unsorted source offset;
    # this catches the in-place permutation corruption that the temp staging
    # buffer is meant to prevent.
    cached.fill_(-1)
    for block in range(batch * heads):
        cached.view(-1, map_size)[block, :source_chunks] = torch.arange(
            source_chunks - 1, -1, -1, dtype=torch.int64, device="cuda"
        )
        values[block, :source_chunks].copy_(torch.stack([
            sources[block][chunk_id, 0]
            for chunk_id in range(source_chunks - 1, -1, -1)
        ]))

    # Move the valid chunks to a different part of the selection.  Invalid
    # tail items must remain zero rather than reading past pinned host memory.
    current.fill_(-1)
    current[:, :, 10 : 10 + source_chunks] = torch.arange(
        source_chunks, dtype=torch.int64, device="cuda"
    )
    run_once()
    for block in range(batch * heads):
        expected = torch.zeros_like(values[block])
        for item, chunk_id in enumerate(reordered.view(-1, map_size)[block].cpu().tolist()):
            if 0 <= chunk_id < source_chunks:
                expected[item].copy_(sources[block][chunk_id, 0])
        torch.testing.assert_close(values[block], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shadowkv_graph_gpu_cache_write_is_replay_safe():
    manager = object.__new__(ShadowKVCacheManager)
    manager.config = SimpleNamespace(
        decode_graph=True,
        shadowkv_storage="gpu_cache",
    )
    manager.num_kv_heads = 2
    manager.head_dim = 4
    manager.device = torch.device("cuda")
    manager._shadow_decode_workspaces = {
        0: {
            "recent_k": torch.zeros(2, 4, 2, 4, dtype=torch.float16, device="cuda"),
            "recent_v": torch.zeros(2, 4, 2, 4, dtype=torch.float16, device="cuda"),
            "prompt_lens": torch.tensor([3, 4], dtype=torch.int32, device="cuda"),
        }
    }
    manager.layer_batch_state = SimpleNamespace(
        context_lens=torch.tensor([4, 6], dtype=torch.int32, device="cuda")
    )
    try:
        set_context(is_prefill=False)
        k = torch.arange(16, device="cuda", dtype=torch.float16).reshape(2, 2, 4)
        v = -k
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            manager.save_rope_kv_if_needed(0, k, v)
        k_next = k + 100
        v_next = v - 100
        manager.layer_batch_state.context_lens.copy_(
            torch.tensor([5, 7], dtype=torch.int32, device="cuda")
        )
        k.copy_(k_next)
        v.copy_(v_next)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            manager._shadow_decode_workspaces[0]["recent_k"][0, 1], k_next[0]
        )
        torch.testing.assert_close(
            manager._shadow_decode_workspaces[0]["recent_v"][1, 2], v_next[1]
        )
    finally:
        reset_context()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shadowkv_eager_gpu_cache_updates_only_the_new_token_in_authoritative_cache():
    """Catch stale authoritative GPU-cache rows after removing workspace copies."""
    manager = object.__new__(ShadowKVCacheManager)
    manager.config = SimpleNamespace(
        decode_graph=False,
        shadowkv_storage="gpu_cache",
        shadowkv_chunk_size=8,
    )
    manager.num_kv_heads = 2
    manager.head_dim = 4
    manager.device = torch.device("cuda")
    manager._shadow_entries = {}
    manager._shadow_active_rows = [0, 1]
    manager.layer_batch_state = SimpleNamespace(
        req_indices=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        context_lens=torch.tensor([4, 6], dtype=torch.int32, device="cuda"),
    )
    for row in (0, 1):
        manager._shadow_entries[(row, 0)] = {
            "gpu_rope_k": torch.zeros(
                8, 2, 4, dtype=torch.float16, device="cuda"
            ),
            "gpu_v": torch.zeros(8, 2, 4, dtype=torch.float16, device="cuda"),
            "rope_k": None,
            "v": None,
            "filled": 0,
        }
    manager._write_shadow = lambda *args, **kwargs: None
    k = torch.arange(16, dtype=torch.float16, device="cuda").reshape(2, 2, 4)
    v = -k
    try:
        set_context(is_prefill=False)
        manager.save_rope_kv_if_needed(0, k, v)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            manager._shadow_entries[(0, 0)]["gpu_rope_k"][3], k[0]
        )
        torch.testing.assert_close(
            manager._shadow_entries[(1, 0)]["gpu_v"][5], v[1]
        )
    finally:
        reset_context()


def test_shadowkv_host_pointer_publication_is_cached_until_host_storage_grows():
    manager = object.__new__(ShadowKVCacheManager)
    manager._shadow_entries = {}
    manager._shadow_host_pointer_signatures = {}
    manager._shadow_host_gather = SimpleNamespace()
    manager._shadow_offset_gather = None
    manager.num_kv_heads = 2
    calls = []

    def set_host_pointers(sources, destination):
        calls.append((tuple(id(source) for source in sources), destination.data_ptr()))

    manager._shadow_host_gather.set_host_pointers = set_host_pointers
    for row in (0, 1):
        manager._shadow_entries[(row, 0)] = {
            "rope_k": torch.empty(4, 2, 4),
            "v": torch.empty(4, 2, 4),
            "u": torch.empty(4, 8),
        }
    workspace = {
        "host_k_ptrs": torch.empty(2, dtype=torch.int64),
        "host_v_ptrs": torch.empty(2, dtype=torch.int64),
        "host_u_ptrs": torch.empty(2, dtype=torch.int64),
        "head_host_u_ptrs": torch.empty(4, dtype=torch.int64),
    }

    manager._refresh_host_pointers(0, [0, 1], workspace)
    manager._refresh_host_pointers(0, [0, 1], workspace)
    assert len(calls) == 4

    manager._shadow_entries[(1, 0)]["v"] = torch.empty(8, 2, 4)
    manager._refresh_host_pointers(0, [0, 1], workspace)
    assert len(calls) == 8
