import json
from pathlib import Path

import pytest
import torch

from sparse_frontier.modelling.attention.query_pool import (
    CaptureRecord,
    QueryPoolExpectations,
    QueryPoolLayer,
    QueryPoolManifest,
    balanced_empirical_weights,
    finalize_capture,
    load_query_pool,
    select_response_coreset,
    validate_no_split_overlap,
    write_query_pool,
)


REVISION = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"


def _manifest(**overrides):
    values = {
        "schema_version": 1,
        "model_id": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "model_revision": REVISION,
        "num_layers": 2,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 8,
        "tp_size": 1,
        "rope_type": "llama3",
        "rope_parameters": {"factor": 8.0, "rope_theta": 500000.0},
        "representation": "prompt_origin_aligned_pre_scale",
        "attention_scale": 8**-0.5,
        "max_decode_offset": 1,
        "pool_size_per_kv_group": 4,
        "coreset_sizes": [2],
        "capture_split": "task_index:0-4",
        "source_manifest_sha256": "a" * 64,
        "created_with_git_commit": "deadbeef",
    }
    values.update(overrides)
    return QueryPoolManifest(**values)


def _expectations(**overrides):
    values = {
        "model_id": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "model_revision": REVISION,
        "num_layers": 2,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 8,
        "tp_size": 1,
        "rope_type": "llama3",
        "rope_parameters": {"factor": 8.0, "rope_theta": 500000.0},
        "representation": "prompt_origin_aligned_pre_scale",
        "attention_scale": 8**-0.5,
    }
    values.update(overrides)
    return QueryPoolExpectations(**values)


def _layer(seed):
    generator = torch.Generator().manual_seed(seed)
    centroid = torch.randn(2, 2, 8, generator=generator, dtype=torch.float32)
    queries = torch.randn(2, 2, 8, generator=generator).to(torch.bfloat16)
    weights = torch.full((2, 2), 0.5, dtype=torch.float32)
    return QueryPoolLayer(
        centroid_by_horizon=centroid,
        coreset_queries={2: queries},
        coreset_weights={2: weights},
    )


def test_manifest_json_round_trip_is_strict():
    manifest = _manifest()
    assert QueryPoolManifest.from_dict(manifest.to_dict()) == manifest
    payload = manifest.to_dict()
    payload["unexpected"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        QueryPoolManifest.from_dict(payload)
    payload = manifest.to_dict()
    payload["schema_version"] = 2
    with pytest.raises(ValueError, match="schema"):
        QueryPoolManifest.from_dict(payload)


def test_write_and_load_online_pool_without_full_queries(tmp_path, monkeypatch):
    output = tmp_path / "pool"
    write_query_pool(_manifest(), [_layer(1), _layer(2)], output)

    calls = []
    original_load = torch.load

    def recording_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    pool = load_query_pool(output, _expectations(), tp_rank=0)

    assert len(pool.layers) == 2
    assert pool.layers[0].centroid_by_horizon.shape == (2, 2, 8)
    assert pool.layers[0].full_queries is None
    assert all(call.get("weights_only") is True for call in calls)
    assert (output / "manifest.json").is_file()
    assert (output / "layer_000_rank_000.pt").is_file()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_revision", "wrong"),
        ("attention_scale", 1.0),
        ("rope_type", "linear"),
        ("tp_size", 2),
        ("head_dim", 16),
    ],
)
def test_metadata_mismatch_fails_closed(tmp_path, field, value):
    output = tmp_path / "pool"
    write_query_pool(_manifest(), [_layer(1), _layer(2)], output)
    with pytest.raises(ValueError, match=field.replace("_", " ") + "|" + field):
        load_query_pool(output, _expectations(**{field: value}), tp_rank=0)


def test_missing_and_corrupt_shards_fail_closed(tmp_path):
    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "manifest.json").write_text(json.dumps(_manifest().to_dict()))
    with pytest.raises((FileNotFoundError, ValueError), match="layer|shard"):
        load_query_pool(missing, _expectations(), tp_rank=0)

    corrupt = tmp_path / "corrupt"
    write_query_pool(_manifest(), [_layer(1), _layer(2)], corrupt)
    (corrupt / "layer_001_rank_000.pt").write_bytes(b"not a tensor archive")
    with pytest.raises(ValueError, match="corrupt|load|shard"):
        load_query_pool(corrupt, _expectations(), tp_rank=0)


def test_tp_rank_outside_manifest_fails(tmp_path):
    output = tmp_path / "pool"
    write_query_pool(_manifest(), [_layer(1), _layer(2)], output)
    with pytest.raises(ValueError, match="TP rank"):
        load_query_pool(output, _expectations(), tp_rank=1)


def test_balanced_weights_give_each_q_head_equal_mass():
    query_heads = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    strata = torch.tensor([0, 0, 1, 0, 1, 0, 0, 1, 2, 0])
    weights = balanced_empirical_weights(query_heads, strata)
    assert weights.sum().item() == pytest.approx(1.0)
    for head in range(4):
        assert weights[query_heads == head].sum().item() == pytest.approx(0.25)
    assert weights[(query_heads == 0) & (strata == 0)].sum().item() == pytest.approx(
        weights[(query_heads == 0) & (strata == 1)].sum().item()
    )


def test_response_coreset_is_deterministic_nonnegative_and_matches_mean():
    queries = torch.arange(48, dtype=torch.float32).view(6, 8)
    features = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.9, 1.0], [1.0, 1.0], [0.4, 0.5], [0.6, 0.5]],
        dtype=torch.float64,
    )
    weights = torch.tensor([1, 1, 2, 2, 1, 1], dtype=torch.float64)
    first = select_response_coreset(queries, features, weights, size=4, seed=43)
    second = select_response_coreset(queries, features, weights, size=4, seed=43)
    assert torch.equal(first.indices, second.indices)
    torch.testing.assert_close(first.weights, second.weights)
    assert first.indices.unique().numel() == 4
    assert torch.all(first.weights >= 0)
    assert first.weights.sum().item() == pytest.approx(1.0)
    target = (weights / weights.sum()) @ features
    approximation = first.weights.double() @ features[first.indices]
    assert torch.linalg.vector_norm(target - approximation).item() < 0.05


def test_split_overlap_is_detected_by_sequence_hash_and_task_index():
    records = [
        CaptureRecord(1, 0, "calibration"),
        CaptureRecord(2, 1, "calibration"),
        CaptureRecord(3, 5, "validation"),
    ]
    validate_no_split_overlap(records)
    with pytest.raises(ValueError, match="sequence|overlap"):
        validate_no_split_overlap(records + [CaptureRecord(1, 8, "validation")])
    with pytest.raises(ValueError, match="task_index|overlap"):
        validate_no_split_overlap(records + [CaptureRecord(4, 1, "validation")])


def _write_raw_step(path: Path, manifest: QueryPoolManifest, offset: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    payload = {
        "prompt_origin_queries": torch.randn(
            manifest.num_layers,
            manifest.num_q_heads,
            manifest.head_dim,
            generator=generator,
            dtype=torch.float32,
        ),
        "metadata": {
            "decode_offset": offset,
            "prompt_length": 8192,
            "sequence_id_hash": 100 + seed,
            "task": "niah_single",
            "task_index": seed,
            "split": "calibration",
            "stratum_id": 0,
            "tp_rank": 0,
        },
    }
    torch.save(payload, path)


def test_finalize_capture_builds_cumulative_centroids_and_offline_pool(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = _manifest(coreset_sizes=[])
    (raw / "manifest.json").write_text(json.dumps(manifest.to_dict()))
    _write_raw_step(raw / "step_000.pt", manifest, offset=0, seed=0)
    _write_raw_step(raw / "step_001.pt", manifest, offset=1, seed=1)

    output = tmp_path / "pool"
    finalize_capture(raw, output, pool_size=4, coreset_sizes=(), seed=43)
    pool = load_query_pool(
        output, _expectations(), tp_rank=0, include_full_queries=True
    )

    assert pool.manifest.pool_size_per_kv_group == 4
    assert pool.layers[0].centroid_by_horizon.shape == (2, 2, 8)
    assert pool.layers[0].full_queries.shape == (2, 4, 8)
    assert pool.layers[0].full_weights.shape == (2, 4)
    torch.testing.assert_close(pool.layers[0].full_weights.sum(-1), torch.ones(2))
    assert (output / "full" / "layer_000_rank_000.pt").is_file()


def test_finalize_accepts_real_capture_payload_and_ignores_prompt_shards(tmp_path):
    raw = tmp_path / "raw"
    request = raw / "request_0001"
    request.mkdir(parents=True)
    manifest = _manifest(coreset_sizes=[])
    (raw / "manifest.json").write_text(json.dumps(manifest.to_dict()))
    torch.save(
        {"keys": torch.zeros(8, 2, 8), "metadata": {"split": "calibration"}},
        request / "prompt_layer_000_rank_000.pt",
    )
    for offset in range(2):
        torch.save(
            {
                "layer_ids": torch.tensor([0, 1]),
                "prompt_origin_queries": torch.randn(2, 4, 8),
                "post_rope_queries": torch.randn(2, 4, 8),
                "generated_keys": torch.randn(2, 1, 2, 8),
                "metadata": {
                    "decode_offset": offset,
                    "prompt_length": 8,
                    "sequence_id_hash": 1,
                    "task": "niah_single",
                    "task_index": 0,
                    "split": "calibration",
                    "stratum_id": 0,
                    "tp_rank": 0,
                },
            },
            request / f"step_{offset:03d}_rank_000.pt",
        )
    output = tmp_path / "pool"
    finalize_capture(raw, output, pool_size=4, coreset_sizes=(), seed=43)
    pool = load_query_pool(output, _expectations(), 0, include_full_queries=True)
    assert len(pool.layers) == 2


def test_query_pool_cli_inspect_emits_machine_readable_json(tmp_path, capsys):
    from sparse_frontier.query_pool_cli import run_cli

    output = tmp_path / "pool"
    write_query_pool(_manifest(), [_layer(1), _layer(2)], output)
    assert run_cli(["inspect", "--pool", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["num_layers"] == 2
    assert payload["online_query_bytes"] > 0


def test_query_pool_cli_returns_nonzero_for_invalid_artifact(tmp_path, capsys):
    from sparse_frontier.query_pool_cli import run_cli

    assert run_cli(["inspect", "--pool", str(tmp_path / "absent")]) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "error"
