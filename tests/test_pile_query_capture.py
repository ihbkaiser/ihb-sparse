import json

import torch

from sparse_frontier.pile_query_capture import (
    QueryReservoir,
    finalize_vllm_capture_query_pool,
    load_pile_query_pool,
)


def test_priority_reservoir_is_exact_size_and_without_replacement():
    reservoir = QueryReservoir(1, 2, 4, 5, seed=43)
    queries = torch.arange(2 * 20 * 4, dtype=torch.float32).view(2, 20, 4)
    reservoir.update(0, queries, torch.arange(20), sequence_index=7)
    reservoir.complete()
    assert reservoir.queries.shape == (1, 2, 5, 4)
    assert reservoir.position.shape == (1, 2, 5)
    for head in range(2):
        assert len(set(reservoir.position[0, head].tolist())) == 5
        assert set(reservoir.sequence_index[0, head].tolist()) == {7}


def test_schema2_pile_loader_groups_query_heads_by_kv(tmp_path):
    manifest = {
        "schema_version": 2,
        "artifact_type": "pile_empirical_query_pool",
        "model_id": "test/model",
        "model_revision": "a" * 40,
        "tokenizer_revision": "a" * 40,
        "num_layers": 1,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "tp_size": 1,
        "head_dim": 4,
        "gqa_group_size": 2,
        "attention_scale": 0.5,
        "rope_type": "llama3",
        "rope_parameters": {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_theta": 500000.0,
        },
        "representation": "post_qk_norm_post_rope_pre_scale",
        "normalization": {"q_norm": False, "k_norm": False},
        "dataset": {"dataset": "pile", "config": "default", "split": "train"},
        "sampling": {
            "method": "independent_uniform_priority_reservoir",
            "without_replacement": True,
            "samples_per_layer_query_head": 3,
            "seed": 43,
        },
        "sequence_records": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    q = torch.randn(4, 3, 4, dtype=torch.bfloat16)
    payload = {
        "queries": q,
        "query_head_id": torch.arange(4, dtype=torch.int16)[:, None].expand(4, 3),
        "kv_head_id": torch.tensor([[0] * 3, [0] * 3, [1] * 3, [1] * 3], dtype=torch.int16),
        "position": torch.arange(3, dtype=torch.int32).repeat(4, 1),
        "sequence_index": torch.zeros(4, 3, dtype=torch.int32),
        "weights": torch.full((4, 3), 1 / 3, dtype=torch.float32),
    }
    torch.save(payload, tmp_path / "layer_000.pt")
    pool = load_pile_query_pool(tmp_path)
    assert pool.layers[0].queries_by_kv_head.shape == (2, 6, 4)
    assert torch.allclose(pool.layers[0].weights_by_kv_head.sum(-1), torch.ones(2))
    assert pool.layers[0].query_head_ids_by_kv_head[1].tolist() == [2, 2, 2, 3, 3, 3]


def test_vllm_capture_finalizer_preserves_origin_positions_and_gqa_groups(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    raw_manifest = {
        "model_id": "test/model",
        "model_revision": "a" * 40,
        "num_layers": 1,
        "num_q_heads": 2,
        "num_kv_heads": 1,
        "tp_size": 1,
        "head_dim": 2,
        "attention_scale": 2**-0.5,
        "rope_type": "llama3",
        "rope_parameters": {"rope_theta": 500000.0},
        "representation": "post_qk_norm_post_rope_pre_scale",
    }
    (raw / "manifest.json").write_text(json.dumps(raw_manifest), encoding="utf-8")
    for offset in range(3):
        torch.save(
            {
                "layer_ids": torch.tensor([0]),
                "prompt_origin_queries": torch.tensor(
                    [[[offset, offset + 1], [offset + 2, offset + 3]]], dtype=torch.bfloat16
                ),
                "metadata": {
                    "split": "calibration",
                    "decode_offset": offset,
                    "sequence_id_hash": 100 + offset,
                },
            },
            raw / f"step_{offset:03d}_rank_000.pt",
        )
    output = finalize_vllm_capture_query_pool(
        raw, tmp_path / "pool", samples_per_head=2, seed=43
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_type"] == "vllm_empirical_query_pool"
    pool = load_pile_query_pool(output)
    layer = pool.layers[0]
    assert layer.queries_by_kv_head.shape == (1, 4, 2)
    assert set(layer.positions_by_kv_head[0].tolist()) <= {0, 1, 2}
    assert layer.query_head_ids_by_kv_head[0].tolist().count(0) == 2
    assert layer.query_head_ids_by_kv_head[0].tolist().count(1) == 2


def test_vllm_capture_finalizer_can_freeze_one_calibration_task_stratum(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    raw_manifest = {
        "model_id": "test/model",
        "model_revision": "a" * 40,
        "num_layers": 1,
        "num_q_heads": 2,
        "num_kv_heads": 1,
        "tp_size": 1,
        "head_dim": 2,
        "attention_scale": 2**-0.5,
        "rope_type": "llama3",
        "rope_parameters": {"rope_theta": 500000.0},
        "representation": "post_qk_norm_post_rope_pre_scale",
    }
    (raw / "manifest.json").write_text(json.dumps(raw_manifest), encoding="utf-8")
    for offset, task in enumerate(("keep", "drop")):
        torch.save(
            {
                "layer_ids": torch.tensor([0]),
                "prompt_origin_queries": torch.tensor(
                    [[[offset, offset + 1], [offset + 2, offset + 3]]], dtype=torch.bfloat16
                ),
                "metadata": {
                    "split": "calibration",
                    "task": task,
                    "decode_offset": offset,
                    "sequence_id_hash": 100 + offset,
                },
            },
            raw / f"step_{offset:03d}_rank_000.pt",
        )
    output = finalize_vllm_capture_query_pool(
        raw, tmp_path / "pool", samples_per_head=1, seed=43, tasks=("keep",)
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sampling"]["task_filter"] == ["keep"]
    pool = load_pile_query_pool(output)
    assert pool.layers[0].positions_by_kv_head[0].tolist() == [0, 0]
