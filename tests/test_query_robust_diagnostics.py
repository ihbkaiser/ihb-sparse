import torch

from sparse_frontier.query_robust_diagnostics import (
    apply_llama3_rope,
    balanced_task_head_centroid,
    build_chunk_summaries,
    retained_attention_mass,
    select_gqa_chunks,
)


def test_llama3_prompt_origin_rotation_is_compositional():
    query = torch.randn(3, 128, dtype=torch.float64)
    kwargs = {
        "base": 500000.0,
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    direct = apply_llama3_rope(query, 8065, **kwargs)
    composed = apply_llama3_rope(
        apply_llama3_rope(query, 2, **kwargs), 8063, **kwargs
    )
    torch.testing.assert_close(composed, direct, rtol=1e-10, atol=1e-10)


def test_empirical_tangent_recovers_high_logmass_chunk_missed_by_mean_key():
    # Full chunks 0 and 4 are mandatory. Uniform summaries route chunk 2
    # because its mean logit is largest, while the empirical tangent exposes
    # the single high-logit token in chunk 1.
    keys = torch.tensor(
        [0.0, 0.0, 5.0, -5.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    ).view(10, 1, 1)
    query = torch.ones(1, 1)
    centroid = torch.ones(1, 1)

    uniform_key, uniform_bias, valid = build_chunk_summaries(
        keys, centroid, chunk_size=2, mode="uniform", scale=1.0
    )
    tangent_key, tangent_bias, _ = build_chunk_summaries(
        keys, centroid, chunk_size=2, mode="centroid_entropy", scale=1.0
    )
    uniform_selected = select_gqa_chunks(
        query, uniform_key, uniform_bias, valid, token_budget=6, chunk_size=2, scale=1.0
    )
    tangent_selected = select_gqa_chunks(
        query, tangent_key, tangent_bias, valid, token_budget=6, chunk_size=2, scale=1.0
    )

    assert torch.where(uniform_selected[0])[0].tolist() == [0, 2, 4]
    assert torch.where(tangent_selected[0])[0].tolist() == [0, 1, 4]
    uniform_mass = retained_attention_mass(
        query, keys, uniform_selected, valid, scale=1.0
    )
    tangent_mass = retained_attention_mass(
        query, keys, tangent_selected, valid, scale=1.0
    )
    assert tangent_mass.item() > uniform_mass.item()


def test_gqa_selection_is_shared_within_group_and_ties_choose_lower_chunk():
    query = torch.zeros(4, 2)
    summary_key = torch.zeros(4, 1, 2)
    summary_bias = torch.zeros(4, 1)
    valid = torch.tensor([2, 2, 2, 1])
    selected = select_gqa_chunks(
        query,
        summary_key,
        summary_bias,
        valid,
        token_budget=5,
        chunk_size=2,
        scale=1.0,
    )
    # Sink chunk 0, recent full chunk 2, and partial chunk 3 consume all five.
    assert selected.shape == (1, 4)
    assert torch.where(selected[0])[0].tolist() == [0, 2, 3]


def test_empirical_centroid_balances_tasks_before_samples():
    queries = torch.tensor([[0.0], [0.0], [0.0], [10.0]])
    task_ids = ["a", "a", "a", "b"]
    query_head_ids = torch.zeros(4, dtype=torch.long)
    centroid = balanced_task_head_centroid(queries, task_ids, query_head_ids)
    torch.testing.assert_close(centroid, torch.tensor([5.0], dtype=torch.float64))
