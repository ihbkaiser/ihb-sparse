import json

import pytest
import torch

from sparse_frontier.modelling.attention.prompt_query_source import (
    load_prompt_query_support,
)


def _write_shard(root, request_id=7, layer=3, q_heads=4, samples=2, dim=6):
    request = root / f"request_{request_id:016x}"
    request.mkdir(parents=True)
    torch.save(
        {
            "prompt_pre_rope_queries": torch.arange(
                q_heads * samples * dim, dtype=torch.float32
            ).view(q_heads, samples, dim).to(torch.bfloat16),
            "prompt_positions": torch.tensor(
                [[0, 2], [1, 3], [0, 1], [2, 3]], dtype=torch.int32
            ),
            "layer_idx": layer,
            "representation": "q_norm_pre_rope_pre_scale",
            "sampling": {
                "method": "uniform_priority_reservoir",
                "samples_per_q_head": samples,
                "seed": 1043,
            },
            "metadata": {
                "sequence_id_hash": request_id,
                "prompt_length": 4,
            },
        },
        request / f"prompt_queries_layer_{layer:03d}_rank_000.pt",
    )
    return request


def test_prompt_query_support_groups_gqa_and_assigns_equal_weights(tmp_path):
    _write_shard(tmp_path)
    support = load_prompt_query_support(
        tmp_path,
        7,
        3,
        tp_rank=0,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=6,
        prompt_length=4,
        samples_per_q_head=2,
    )
    assert support.queries_by_kv_head.shape == (2, 4, 6)
    assert support.positions_by_kv_head.shape == (2, 4)
    torch.testing.assert_close(
        support.weights_by_kv_head,
        torch.full((2, 4), 0.25),
    )
    torch.testing.assert_close(
        support.queries_by_kv_head[1, 0],
        torch.arange(24, 30, dtype=torch.float32).to(torch.bfloat16),
    )


def test_prompt_query_support_fails_closed_on_missing_or_generated_fields(tmp_path):
    _write_shard(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_prompt_query_support(
            tmp_path,
            8,
            3,
            tp_rank=0,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=6,
            prompt_length=4,
        )
    path = tmp_path / "request_0000000000000007" / "prompt_queries_layer_003_rank_000.pt"
    payload = torch.load(path, weights_only=True)
    payload["generated_queries"] = torch.zeros(1)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="fields"):
        load_prompt_query_support(
            tmp_path,
            7,
            3,
            tp_rank=0,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=6,
            prompt_length=4,
        )


def test_prompt_query_support_validates_capture_manifest_when_expected(tmp_path):
    _write_shard(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "test/model",
                "model_revision": "rev-a",
                "num_layers": 4,
                "num_q_heads": 4,
                "num_kv_heads": 2,
                "head_dim": 6,
                "tp_size": 1,
                "rope_type": "llama3",
                "rope_parameters": {"rope_theta": 500000.0},
                "attention_scale": 6**-0.5,
                "representation": "prompt_origin_aligned_pre_scale",
            }
        ),
        encoding="utf-8",
    )
    expected = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    load_prompt_query_support(
        tmp_path,
        7,
        3,
        tp_rank=0,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=6,
        prompt_length=4,
        expected_manifest=expected,
    )
    expected["model_revision"] = "rev-b"
    with pytest.raises(ValueError, match="manifest"):
        load_prompt_query_support(
            tmp_path,
            7,
            3,
            tp_rank=0,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=6,
            prompt_length=4,
            expected_manifest=expected,
        )
