from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sparsevllm.engine.cache_manager.query_robust import (
    build_query_robust_page_summaries,
    load_query_robust_asset,
    score_query_robust_pages_reference,
    solve_query_robust_page,
)
from sparsevllm.kernels.triton.query_robust_summary import (
    build_query_robust_page_summaries_from_cache,
)
from sparsevllm.method_registry import is_paged_sparse_method, normalize_sparse_method


def _load_selector_module():
    path = Path(__file__).parents[1] / "tools/query_robust/select_vertices.py"
    spec = importlib.util.spec_from_file_location("query_robust_select_vertices", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mass(query: torch.Tensor, keys: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.logsumexp(scale * (query @ keys.transpose(0, 1)), dim=-1)


def _bound(query: torch.Tensor, landmark: torch.Tensor, bias: torch.Tensor, scale: float) -> torch.Tensor:
    return scale * (query @ landmark.float()) + bias.float()


def test_query_robust_solver_certifies_interior_hull_and_bf16_storage():
    torch.manual_seed(7)
    keys = torch.randn(16, 8)
    vertices = torch.randn(12, 8)
    summary = solve_query_robust_page(
        keys,
        vertices,
        scale=8.0**-0.5,
        solver_iters=48,
        solver_lr=0.25,
    )

    vertex_errors = _mass(vertices, keys, 8.0**-0.5) - _bound(
        vertices, summary.landmark, summary.bias, 8.0**-0.5
    )
    assert torch.all(vertex_errors >= -1e-5)
    assert torch.all(vertex_errors <= summary.epsilon + 1e-5)
    assert float(summary.dual) <= float(summary.epsilon) + 1e-5
    assert float(summary.gap) >= -1e-5
    assert torch.allclose(vertex_errors, summary.errors, atol=1e-5, rtol=1e-5)

    coefficients = torch.distributions.Dirichlet(
        torch.ones(vertices.shape[0])
    ).sample((64,))
    interior = coefficients @ vertices
    interior_errors = _mass(interior, keys, 8.0**-0.5) - _bound(
        interior, summary.landmark, summary.bias, 8.0**-0.5
    )
    assert torch.all(interior_errors >= -1e-5)
    assert torch.all(interior_errors <= summary.epsilon + 1e-5)


def test_query_robust_negative_control_is_outside_hull():
    vertices = torch.tensor([[-1.0], [1.0]])
    keys = torch.tensor([[-2.0], [2.0]])
    summary = solve_query_robust_page(
        keys,
        vertices,
        scale=1.0,
        solver_iters=64,
        solver_lr=0.25,
    )
    inside = torch.tensor([[-0.5], [0.0], [0.5]])
    outside = torch.tensor([[-1.5], [1.5]])
    inside_error = _mass(inside, keys, 1.0) - _bound(
        inside, summary.landmark, summary.bias, 1.0
    )
    outside_error = _mass(outside, keys, 1.0) - _bound(
        outside, summary.landmark, summary.bias, 1.0
    )
    assert torch.all(inside_error <= summary.epsilon + 1e-5)
    assert torch.any(outside_error > summary.epsilon + 1e-3)


def test_query_robust_uniform_baseline_is_mean_key_and_log_page_size():
    keys = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 5.0], [0.0, -2.0]])
    vertices = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    summary = solve_query_robust_page(
        keys,
        vertices,
        scale=0.5,
        solver_iters=24,
        solver_lr=0.25,
        uniform_p=True,
    )
    expected_landmark = keys.mean(dim=0).to(torch.bfloat16)
    assert torch.equal(summary.landmark, expected_landmark)
    assert torch.allclose(summary.bias, torch.log(torch.tensor(4.0)), atol=1e-6)


def test_query_robust_batched_builder_matches_single_page_reference():
    torch.manual_seed(19)
    keys = torch.randn(3, 5, 2, 4)
    vertices = torch.randn(2, 7, 4)
    batched = build_query_robust_page_summaries(
        keys,
        vertices,
        scale=0.5,
        solver_iters=24,
        solver_lr=0.25,
        uniform_p=False,
    )
    for page in range(3):
        for head in range(2):
            single = solve_query_robust_page(
                keys[page, :, head],
                vertices[head],
                scale=0.5,
                solver_iters=24,
                solver_lr=0.25,
            )
            assert torch.equal(batched.landmark[page, head], single.landmark)
            assert torch.allclose(batched.bias[page, head], single.bias, atol=1e-6)
            assert torch.allclose(batched.epsilon[page, head], single.epsilon, atol=1e-6)
            assert torch.allclose(batched.errors[page, head], single.errors, atol=1e-6)


def test_query_robust_reference_scorer_hoists_bias_and_invalidates_stale_pages():
    query = torch.tensor(
        [[[1.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.0, 0.5]]]
    )
    landmark = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]], [[2.0, 0.0], [0.0, 2.0]]],
        dtype=torch.bfloat16,
    )
    bias = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    epsilon = torch.tensor([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
    valid = torch.tensor([[True, True], [True, False], [True, True]])
    slots = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    scores = score_query_robust_pages_reference(
        query,
        landmark,
        bias,
        epsilon,
        valid,
        slots,
        scale=1.0,
        alpha=0.5,
    )
    expected = torch.tensor(
        [
            max(1.0 + 0.1 + 0.005, 1.0 + 0.2 + 0.01),
            float("inf"),
            max(2.0 + 0.5 + 0.025, 2.0 + 0.6 + 0.03),
        ]
    )
    scores = scores.squeeze(0)
    assert torch.equal(torch.isinf(scores), torch.isinf(expected))
    assert torch.allclose(scores[~torch.isinf(scores)], expected[~torch.isinf(expected)])


def test_query_robust_asset_loader_tp_slices_and_rejects_invalid_partition(tmp_path):
    vertices = torch.arange(2 * 4 * 3 * 5, dtype=torch.float32).view(2, 4, 3, 5).to(torch.bfloat16)
    valid = torch.full((2, 4), 3, dtype=torch.int32)
    path = tmp_path / "vertices.pt"
    torch.save(
        {
            "vertices": vertices,
            "num_valid_vertices": valid,
            "meta": {"model_id": "model", "tp_world_size": 2},
        },
        path,
    )
    asset = load_query_robust_asset(
        path,
        num_layers=2,
        global_num_kv_heads=4,
        head_dim=5,
        num_vertices=3,
        tensor_parallel_rank=1,
        tensor_parallel_size=2,
        expected_model_id="/models/model",
    )
    assert tuple(asset.vertices.shape) == (2, 2, 3, 5)
    assert torch.equal(asset.vertices, vertices[:, 2:].float())
    try:
        load_query_robust_asset(
            path,
            num_layers=2,
            global_num_kv_heads=4,
            head_dim=5,
            num_vertices=3,
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
        )
    except ValueError as error:
        assert "TP partition" in str(error)
    else:
        raise AssertionError("asset with a mismatched TP partition was accepted")

    npz_path = tmp_path / "vertices.npz"
    np.savez_compressed(
        npz_path,
        vertices=vertices.view(torch.uint16).numpy(),
        num_valid_vertices=valid.numpy(),
        meta=json.dumps({"model_id": "model", "tp_world_size": 2}),
    )
    npz_asset = load_query_robust_asset(
        npz_path,
        num_layers=2,
        global_num_kv_heads=4,
        head_dim=5,
        num_vertices=3,
        tensor_parallel_rank=0,
        tensor_parallel_size=2,
        expected_model_id="model",
    )
    assert torch.equal(npz_asset.vertices, vertices[:, :2].float())


def test_query_robust_vertex_selector_repeats_real_support_points():
    selector = _load_selector_module()
    queries = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    vertices, valid_count = selector.select_group(queries, num_vertices=6)
    assert valid_count == 2
    assert torch.all(torch.isfinite(vertices))
    assert set(map(tuple, vertices.tolist())) == {(1.0, 0.0), (-1.0, 0.0)}


def test_query_robust_selector_uses_gqa_qfilter_support_extrema():
    selector = _load_selector_module()
    queries = torch.tensor(
        [
            [5.0, 0.0],
            [4.0, 1.0],
            [-4.0, 0.0],
            [-3.0, -1.0],
            [6.0, 1.0],
            [5.0, -1.0],
            [-5.0, 1.0],
            [-4.0, -1.0],
        ]
    )
    q_head_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    vertices, valid_count = selector.select_group(
        queries,
        num_vertices=3,
        q_head_ids=q_head_ids,
    )
    assert valid_count == 3
    assert float(vertices[:, 0].max()) >= 6.0
    assert float(vertices[:, 0].min()) <= -5.0


def test_query_robust_method_is_registered_as_paged_runtime_contract():
    assert normalize_sparse_method("qr") == "query_robust"
    assert is_paged_sparse_method("query_robust")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_query_robust_fused_cache_summary_matches_gathered_and_torch_reference():
    torch.manual_seed(23)
    device = torch.device("cuda")
    page_size = 16
    cache = torch.randn(11 * page_size, 2, 16, device=device, dtype=torch.bfloat16)
    page_slots = torch.tensor([7, 1, 10], device=device, dtype=torch.int32)
    active = torch.tensor([True, True, False], device=device)
    vertices = torch.randn(2, 16, 16, device=device, dtype=torch.float32)

    token_offsets = torch.arange(page_size, device=device, dtype=torch.long)
    token_slots = (
        page_slots.to(torch.long)[:, None] * page_size + token_offsets[None, :]
    )
    gathered = cache.index_select(0, token_slots.reshape(-1)).view(
        page_slots.numel(), page_size, 2, 16
    )

    from sparsevllm.kernels.triton.query_robust_summary import (
        build_query_robust_page_summaries,
    )

    expected = build_query_robust_page_summaries(
        gathered,
        vertices,
        active,
        scale=0.5,
        solver_iters=4,
        solver_lr=0.25,
        uniform_p=False,
    )
    actual = build_query_robust_page_summaries_from_cache(
        cache,
        page_slots,
        vertices,
        active,
        page_size=page_size,
        scale=0.5,
        solver_iters=4,
        solver_lr=0.25,
        uniform_p=False,
    )

    for actual_value, expected_value in zip(actual, expected):
        assert torch.allclose(actual_value, expected_value, atol=2e-2, rtol=2e-2)
