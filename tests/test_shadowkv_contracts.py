from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from benchmark.kvpress_ruler.evaluate import _score
from sparsevllm.engine.cache_manager.shadowkv import ShadowKVCacheManager
from sparsevllm.method_registry import normalize_sparse_method
from sparsevllm.utils.context import reset_context, set_context


def test_shadowkv_alias_and_ruler_scorer_contract():
    assert normalize_sparse_method("shadow-kv") == "shadowkv"
    assert _score("qa_single_1", "The answer is Paris.", ["Paris"]) == 1.0
    assert _score("fwe_1", "alpha beta", ["alpha", "missing"]) == 0.5


def test_shadowkv_factorizes_flattened_keys_with_independent_shape_oracle():
    manager = object.__new__(ShadowKVCacheManager)
    manager._shadow_entries = {}
    manager.config = SimpleNamespace(
        shadowkv_chunk_size=8,
        shadowkv_local_chunks=4,
        shadowkv_sparse_budget=16,
        shadowkv_outlier_chunks=1,
        shadowkv_rank=4,
    )
    manager.num_kv_heads = 2
    manager.head_dim = 4
    manager.device = torch.device("cpu")
    manager.hf_config = SimpleNamespace(dtype=torch.float32)

    entry = manager._entry(0, 0)
    entry["raw_k"] = torch.randn(128, 2, 4)
    entry["rope_k"] = torch.randn(128, 2, 4)
    entry["v"] = torch.randn(128, 2, 4)
    entry["filled"] = 128
    manager._finalize_entry(0, 0, 128)

    assert entry["shadow"] is True
    assert entry["u"].shape == (128, 4)
    assert entry["sv"].shape == (2, 4, 4)
    assert entry["landmarks"].shape[0] == manager.num_kv_heads
    assert entry["raw_k"] is None


def test_shadowkv_zero_outlier_configuration_keeps_per_head_metadata_ranked():
    manager = object.__new__(ShadowKVCacheManager)
    manager._shadow_entries = {}
    manager.config = SimpleNamespace(
        shadowkv_chunk_size=8,
        shadowkv_local_chunks=1,
        shadowkv_sparse_budget=8,
        shadowkv_outlier_chunks=0,
        shadowkv_rank=4,
    )
    manager.num_kv_heads = 2
    manager.head_dim = 4
    manager.device = torch.device("cpu")
    manager.hf_config = SimpleNamespace(dtype=torch.float32)

    entry = manager._entry(0, 0)
    entry["raw_k"] = torch.randn(64, 2, 4)
    entry["rope_k"] = torch.randn(64, 2, 4)
    entry["v"] = torch.randn(64, 2, 4)
    entry["filled"] = 64
    manager._finalize_entry(0, 0, 64)

    assert tuple(entry["outlier_chunks"].shape) == (2, 0)


def test_shadowkv_factorizes_pre_rope_keys_not_post_rope_keys():
    manager = object.__new__(ShadowKVCacheManager)
    manager._shadow_entries = {}
    manager.config = SimpleNamespace(
        shadowkv_chunk_size=8,
        shadowkv_local_chunks=1,
        shadowkv_sparse_budget=8,
        shadowkv_outlier_chunks=1,
        shadowkv_rank=8,
    )
    manager.num_kv_heads = 2
    manager.head_dim = 4
    manager.device = torch.device("cpu")
    manager.hf_config = SimpleNamespace(dtype=torch.float32)

    torch.manual_seed(20260907)
    raw_k = torch.randn(64, 2, 4)
    post_rope_k = raw_k.roll(shifts=1, dims=-1)
    entry = manager._entry(0, 0)
    entry["raw_k"] = raw_k.clone()
    entry["rope_k"] = post_rope_k
    entry["v"] = torch.randn_like(raw_k)
    entry["filled"] = raw_k.shape[0]
    manager._finalize_entry(0, 0, raw_k.shape[0])

    reconstructed = torch.einsum(
        "pr,hrd->phd",
        entry["u"],
        entry["sv"],
    )
    torch.testing.assert_close(reconstructed, raw_k, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(reconstructed, post_rope_k, atol=1e-4, rtol=1e-4)


def test_shadowkv_batched_factorization_preserves_exact_reconstruction():
    manager = object.__new__(ShadowKVCacheManager)
    manager._shadow_entries = {}
    manager.config = SimpleNamespace(
        shadowkv_chunk_size=8,
        shadowkv_local_chunks=1,
        shadowkv_sparse_budget=8,
        shadowkv_outlier_chunks=1,
        shadowkv_rank=8,
    )
    manager.num_kv_heads = 2
    manager.head_dim = 4
    manager.device = torch.device("cpu")
    manager.hf_config = SimpleNamespace(dtype=torch.float32)

    torch.manual_seed(17)
    originals = []
    for row in (0, 1):
        raw_k = torch.randn(64, 2, 4)
        originals.append(raw_k)
        entry = manager._entry(row, 0)
        entry["raw_k"] = raw_k.clone()
        entry["rope_k"] = torch.randn_like(raw_k)
        entry["v"] = torch.randn_like(raw_k)
        entry["filled"] = raw_k.shape[0]

    manager._finalize_entries_batched([0, 1], 0, 64)
    for row, raw_k in enumerate(originals):
        entry = manager._entry(row, 0)
        reconstructed = torch.einsum("pr,hrd->phd", entry["u"], entry["sv"])
        torch.testing.assert_close(reconstructed, raw_k, atol=1e-4, rtol=1e-4)


def test_shadowkv_decode_workspace_uses_row_local_slots():
    manager = object.__new__(ShadowKVCacheManager)
    manager.config = SimpleNamespace(
        shadowkv_chunk_size=8,
        shadowkv_sparse_budget=16,
        shadowkv_outlier_chunks=1,
        shadowkv_local_chunks=1,
        shadowkv_recent_tokens=8,
        shadowkv_rank=4,
        decode_graph=False,
    )
    manager.num_kv_heads = 2
    manager.head_dim = 4
    manager.device = torch.device("cpu")
    manager.hf_config = SimpleNamespace(dtype=torch.float32)
    manager._shadow_decode_workspaces = {}

    workspace = manager._ensure_decode_workspace(0, batch_size=3, context_capacity=32)
    width = int(workspace["active_slots"].shape[1])
    expected = torch.arange(3 * width, dtype=torch.int32).view(3, width)
    torch.testing.assert_close(workspace["active_slots"], expected)
    assert workspace["local_req_indices"].dtype == torch.int32
    torch.testing.assert_close(
        workspace["local_req_indices"], torch.arange(3, dtype=torch.int32)
    )


def test_shadowkv_decode_graph_does_not_force_short_batches_eager():
    """Protect the single graph topology used by both short and long requests."""
    manager = object.__new__(ShadowKVCacheManager)
    assert (
        manager.decode_graph_force_eager_for_batch([], is_long_text=False) is False
    )
    assert (
        manager.decode_graph_force_eager_for_batch([], is_long_text=True) is False
    )


def test_shadowkv_gpu_cache_materializes_exact_prompt_payload():
    manager = object.__new__(ShadowKVCacheManager)
    manager._shadow_entries = {}
    manager.config = SimpleNamespace(
        shadowkv_chunk_size=8,
        shadowkv_local_chunks=1,
        shadowkv_sparse_budget=8,
        shadowkv_outlier_chunks=1,
        shadowkv_rank=8,
        shadowkv_storage="gpu_cache",
        shadowkv_gpu_cache_tokens=80,
    )
    manager.num_kv_heads = 2
    manager.head_dim = 4
    manager.device = torch.device("cpu")
    manager.hf_config = SimpleNamespace(dtype=torch.float32)

    raw_k = torch.randn(64, 2, 4)
    rope_k = torch.randn_like(raw_k)
    values = torch.randn_like(raw_k)
    entry = manager._entry(0, 0)
    entry["raw_k"] = raw_k
    entry["rope_k"] = rope_k
    entry["v"] = values
    entry["filled"] = 64
    manager._finalize_entry(0, 0, 64)

    assert entry["gpu_rope_k"].shape == (80, 2, 4)
    assert entry["gpu_v"].shape == (80, 2, 4)
    torch.testing.assert_close(entry["gpu_rope_k"][:64], rope_k)
    torch.testing.assert_close(entry["gpu_v"][:64], values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shadowkv_gpu_cache_does_not_factorize_unused_payload(monkeypatch):
    """GPU-cache selection must not pay for an unused CPU-shadow SVD."""
    manager = object.__new__(ShadowKVCacheManager)
    manager._shadow_entries = {}
    manager.config = SimpleNamespace(
        shadowkv_chunk_size=8,
        shadowkv_local_chunks=1,
        shadowkv_sparse_budget=8,
        shadowkv_outlier_chunks=1,
        shadowkv_rank=8,
        shadowkv_storage="gpu_cache",
        shadowkv_gpu_cache_tokens=136,
        shadowkv_svd_method="exact",
    )
    manager.num_kv_heads = 2
    manager.head_dim = 4
    manager.device = torch.device("cuda")
    manager.hf_config = SimpleNamespace(dtype=torch.float32)

    raw_k = torch.randn(128, 2, 4)
    rope_k = torch.randn_like(raw_k)
    values = torch.randn_like(raw_k)
    entry = manager._entry(0, 0)
    entry["raw_k"] = raw_k
    entry["rope_k"] = rope_k
    entry["v"] = values
    entry["filled"] = 128

    def fail_svd(*args, **kwargs):
        raise AssertionError("GPU-cache finalization unexpectedly invoked SVD")

    monkeypatch.setattr(torch.linalg, "svd", fail_svd)
    manager._finalize_entry(0, 0, 128)

    assert entry["u"] is None
    assert entry["sv"] is None
    assert entry["sv_column_major"] is None
    assert entry["landmarks"].shape == (2, 7, 4)
    torch.testing.assert_close(entry["gpu_rope_k"][:128].cpu(), rope_k)
    torch.testing.assert_close(entry["gpu_v"][:128].cpu(), values)

    # The GPU metadata path must preserve the CPU reference's per-head
    # landmark/outlier semantics even though it avoids the unused SVD.
    chunks = 8
    context = (
        rope_k[: chunks * 8]
        .view(chunks, 8, manager.num_kv_heads, manager.head_dim)
        .permute(2, 0, 1, 3)
        .contiguous()
    )
    expected_landmarks = context.mean(dim=2)
    expected_outliers = F.cosine_similarity(
        expected_landmarks.unsqueeze(2), context, dim=-1
    ).amin(dim=-1).topk(1, largest=False, dim=-1).indices
    expected_indices = []
    expected_kept = []
    for head in range(manager.num_kv_heads):
        mask = torch.zeros(chunks, dtype=torch.bool)
        mask[expected_outliers[head]] = True
        expected_indices.append(torch.arange(chunks)[~mask])
        expected_kept.append(expected_landmarks[head][~mask])
    torch.testing.assert_close(
        entry["landmark_indices"], torch.stack(expected_indices)
    )
    torch.testing.assert_close(entry["landmarks"], torch.stack(expected_kept))


def test_shadowkv_gpu_cache_stages_chunked_prefill_before_factorization():
    manager = object.__new__(ShadowKVCacheManager)
    manager._shadow_entries = {}
    manager.config = SimpleNamespace(
        shadowkv_storage="gpu_cache",
        shadowkv_gpu_cache_tokens=8,
    )
    manager.num_kv_heads = 2
    manager.head_dim = 4
    manager.device = torch.device("cpu")
    manager.hf_config = SimpleNamespace(dtype=torch.float32)
    manager.layer_batch_state = SimpleNamespace(
        req_indices=torch.tensor([0, 1], dtype=torch.int32),
        context_lens=torch.tensor([3, 2], dtype=torch.int32),
    )

    first_k = torch.randn(5, 2, 4)
    first_v = torch.randn_like(first_k)
    try:
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, 3, 5], dtype=torch.int32),
        )
        manager.save_rope_kv_if_needed(0, first_k, first_v)
        torch.testing.assert_close(manager._entry(0, 0)["gpu_rope_k"][:3], first_k[:3])
        torch.testing.assert_close(manager._entry(1, 0)["gpu_v"][:2], first_v[3:5])

        manager.layer_batch_state.context_lens.copy_(torch.tensor([5, 4]))
        second_k = torch.randn(4, 2, 4)
        second_v = torch.randn_like(second_k)
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
        )
        manager.save_rope_kv_if_needed(0, second_k, second_v)
        torch.testing.assert_close(
            manager._entry(0, 0)["gpu_rope_k"][:5], torch.cat((first_k[:3], second_k[:2]))
        )
        torch.testing.assert_close(
            manager._entry(1, 0)["gpu_v"][:4], torch.cat((first_v[3:5], second_v[2:4]))
        )
    finally:
        reset_context()
