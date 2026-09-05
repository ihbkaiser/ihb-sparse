import hashlib
import json

import pytest


def _write_table1_fixture(tmp_path):
    from sparse_frontier.ruler_protocol import TABLE1_TASKS

    rows = [
        {
            "context": f"context for {task}",
            "question": "question",
            "answer_prefix": "Answer:",
            "answer": ["gold"],
            "task": task,
            "max_new_tokens": 16,
        }
        for task in TABLE1_TASKS
    ]
    rows_path = tmp_path / "ruler-128k.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "protocol": "ruler_128k_table1_recipe",
        "source": {
            "name": "RULER",
            "repository": "hsiehjackson/RULER",
            "revision": "a" * 40,
        },
        "rows_path": rows_path.name,
        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "context_length": 131072,
        "seed": 43,
        "task_set": list(TABLE1_TASKS),
        "samples_per_task": 1,
        "model": {"id": "org/model", "revision": "b" * 40},
        "tokenizer": {
            "id": "org/model",
            "revision": "b" * 40,
            "template_sha256": hashlib.sha256(b"template-v1").hexdigest(),
        },
        "runtime": {
            "dtype": "bfloat16",
            "greedy": True,
            "batch_size": 1,
            "prefill": "dense_exact",
            "max_input_tokens": 131072,
            "max_output_tokens": 128,
            "task_output_caps": {task: 16 for task in TABLE1_TASKS},
        },
        "hardware_policy": {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
            "cpu_offload_gb": 0.0,
            "max_num_batched_tokens": 131200,
        },
    }
    manifest_path = tmp_path / "protocol.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, rows_path


def test_table1_protocol_locks_128k_rows_and_runtime_contract(tmp_path):
    from sparse_frontier.ruler_protocol import TABLE1_TASKS, load_table1_protocol

    manifest_path, _ = _write_table1_fixture(tmp_path)
    contract = load_table1_protocol(manifest_path)

    assert contract.context_length == 131072
    assert contract.seed == 43
    assert contract.task_set == TABLE1_TASKS
    assert [row["task"] for row in contract.rows] == list(TABLE1_TASKS)
    assert contract.runtime["prefill"] == "dense_exact"
    assert contract.runtime["dtype"] == "bfloat16"
    assert contract.hardware_policy["max_num_batched_tokens"] == 131200


def test_table1_protocol_rejects_data_that_no_longer_matches_manifest_sha(tmp_path):
    from sparse_frontier.ruler_protocol import load_table1_protocol

    manifest_path, rows_path = _write_table1_fixture(tmp_path)
    rows_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256"):
        load_table1_protocol(manifest_path)


def test_table1_protocol_accepts_only_a_pile_query_pool(tmp_path):
    from sparse_frontier.ruler_protocol import validate_pile_only_query_pool

    pool = tmp_path / "pile-pool"
    pool.mkdir()
    (pool / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "artifact_type": "pile_empirical_query_pool",
                "dataset": {"dataset": "monology/pile-uncopyrighted", "split": "train"},
            }
        ),
        encoding="utf-8",
    )
    assert validate_pile_only_query_pool(pool) == pool.resolve()

    (pool / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "artifact_type": "vllm_empirical_query_pool"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Pile-only"):
        validate_pile_only_query_pool(pool)


def test_runner_preparation_uses_locked_runtime_and_hardware_policy(tmp_path, monkeypatch):
    from sparse_frontier.ruler_runner import prepare_table1_protocol_run

    manifest_path, _ = _write_table1_fixture(tmp_path)
    pool = tmp_path / "pile-pool"
    pool.mkdir()
    (pool / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "artifact_type": "pile_empirical_query_pool",
                "dataset": {"dataset": "monology/pile-uncopyrighted", "split": "train"},
            }
        ),
        encoding="utf-8",
    )

    prepared = prepare_table1_protocol_run(manifest_path, query_pool_path=pool)

    assert [(spec.method, spec.budget) for spec in prepared.run_specs] == [
        ("dense", None),
        ("quest", 2048),
        ("shadowkv", 2048),
        ("query_robust", 2048),
    ]
    assert prepared.max_input_tokens == 131072
    assert prepared.max_output_tokens == 128
    assert prepared.seed == 43
    assert prepared.tp == 1
    assert __import__("os").environ["SF_MAX_NUM_BATCHED_TOKENS"] == "131200"
    assert __import__("os").environ["SF_GPU_MEMORY_UTILIZATION"] == "0.85"


def test_table1_protocol_validates_model_revision_and_chat_template(tmp_path):
    from sparse_frontier.ruler_protocol import (
        load_table1_protocol,
        validate_table1_model_contract,
        validate_table1_tokenizer_contract,
    )

    manifest_path, _ = _write_table1_fixture(tmp_path)
    contract = load_table1_protocol(manifest_path)
    validate_table1_model_contract(
        contract, {"model_id": "org/model", "model_revision": "b" * 40}
    )
    with pytest.raises(ValueError, match="model revision"):
        validate_table1_model_contract(
            contract, {"model_id": "org/model", "model_revision": "c" * 40}
        )

    tokenizer = type("Tokenizer", (), {"chat_template": "template-v1"})()
    validate_table1_tokenizer_contract(contract, tokenizer)
    with pytest.raises(ValueError, match="template SHA256"):
        validate_table1_tokenizer_contract(
            contract, type("Tokenizer", (), {"chat_template": "different"})()
        )
