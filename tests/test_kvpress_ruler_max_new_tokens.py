import sys
from pathlib import Path

import pytest

import benchmark.kvpress_ruler.evaluate as ruler_evaluate
from benchmark.kvpress_ruler.evaluate import (
    EvalConfig,
    _build_infer_config,
    _build_evaluation_groups,
    _evaluation_timing,
    _infer_max_model_len,
    _load_rows,
    _parse_args,
    _resolve_max_new_tokens,
    _row_prompt,
)


def test_inferred_max_model_len_is_exact_prompt_generation_budget():
    assert _infer_max_model_len([130_944, 130_000], [128, 32]) == 131_072


def test_ruler_throughput_timing_starts_after_startup():
    timing = _evaluation_timing(
        started_at=10.0,
        serving_started_at=40.0,
        finished_at=70.0,
    )

    assert timing["startup_seconds"] == 30.0
    assert timing["serving_elapsed_seconds"] == 30.0
    assert timing["elapsed_seconds"] == 30.0
    assert timing["total_elapsed_seconds"] == 60.0


def test_kvpress_defaults_use_gpu_cache_shadowkv_profile(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--model-path",
            "/models/llama",
            "--output-dir",
            "/results/quest",
            "--sparse-method",
            "quest",
        ],
    )

    config = _parse_args()

    assert config.quest_chunk_size == 16
    assert config.sink_keep_tokens == 0
    assert config.decode_keep_tokens == 2048
    assert config.recent_keep_tokens == 0
    assert (
        config.sink_keep_tokens
        + config.decode_keep_tokens
        + config.recent_keep_tokens
        == 2048
    )
    assert config.shadowkv_sparse_budget == 2048
    assert config.shadowkv_rank == 160
    assert config.shadowkv_chunk_size == 8
    assert config.shadowkv_storage == "gpu_cache"


def test_kvpress_parser_accepts_query_robust_full_ruler_configuration(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--model-path",
            "/models/llama",
            "--output-dir",
            "/results/qr",
            "--dataset-path",
            "/data/ruler-131072.jsonl",
            "--sparse-method",
            "query_robust",
            "--query-robust-vertices-path",
            "/data/qr_vertices.pt",
            "--query-robust-num-vertices",
            "8",
            "--query-robust-chunk-size",
            "16",
            "--decode-keep-tokens",
            "2048",
            "--sink-keep-tokens",
            "0",
            "--recent-keep-tokens",
            "0",
        ],
    )

    config = _parse_args()

    assert config.sparse_method == "query_robust"
    assert config.query_robust_vertices_path == "/data/qr_vertices.pt"
    assert config.query_robust_num_vertices == 8
    assert config.query_robust_chunk_size == 16
    assert (
        config.sink_keep_tokens
        + config.decode_keep_tokens
        + config.recent_keep_tokens
        == 2048
    )


def test_kvpress_parser_propagates_shadowkv_outlier_and_local_chunks(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--model-path",
            "/models/llama",
            "--output-dir",
            "/results/shadowkv",
            "--sparse-method",
            "shadowkv",
            "--shadowkv-sparse-budget",
            "512",
            "--shadowkv-chunk-size",
            "8",
            "--shadowkv-rank",
            "160",
            "--shadowkv-outlier-chunks",
            "48",
            "--shadowkv-local-chunks",
            "4",
            "--shadowkv-storage",
            "cpu",
        ],
    )

    config = _parse_args()
    infer_config = _build_infer_config(config, resolved_max_model_len=32768)

    assert config.shadowkv_outlier_chunks == 48
    assert config.shadowkv_local_chunks == 4
    assert infer_config["shadowkv_outlier_chunks"] == 48
    assert infer_config["shadowkv_local_chunks"] == 4


def test_kvpress_builds_native_query_robust_runtime_config():
    config = EvalConfig(
        model_path="/models/llama",
        output_dir="/results/qr",
        sparse_method="query_robust",
        query_robust_vertices_path="/data/qr_vertices.pt",
        query_robust_num_vertices=8,
        query_robust_chunk_size=16,
        query_robust_solver_iters=24,
        query_robust_solver_lr=0.25,
        query_robust_score_alpha=0.5,
        query_robust_skip_layers=0,
        sink_keep_tokens=0,
        decode_keep_tokens=2048,
        recent_keep_tokens=0,
    )

    infer_config = _build_infer_config(config, resolved_max_model_len=131072)

    assert infer_config["query_robust_vertices_path"] == "/data/qr_vertices.pt"
    assert infer_config["query_robust_solver_iters"] == 24
    assert infer_config["decode_keep_tokens"] == 2048


def test_kvpress_propagates_explicit_max_batched_tokens():
    config = EvalConfig(
        model_path="/models/llama",
        output_dir="/results/shadowkv",
        sparse_method="shadowkv",
        max_batched_tokens=131072,
    )

    infer_config = _build_infer_config(config, resolved_max_model_len=32784)

    assert infer_config["max_num_batched_tokens"] == 131072


def test_ruler_uses_each_row_max_new_tokens_by_default():
    rows = [
        {"task": "niah_single_1", "max_new_tokens": 30},
        {"task": "qa_1", "max_new_tokens": 128},
    ]

    assert _resolve_max_new_tokens(rows, override=None) == [30, 128]


def test_ruler_cli_override_replaces_each_row_budget_explicitly():
    rows = [
        {"task": "niah_single_1", "max_new_tokens": 30},
        {"task": "qa_1", "max_new_tokens": 128},
    ]

    assert _resolve_max_new_tokens(rows, override=64) == [64, 64]


def test_evaluation_groups_keep_task_rows_together_with_matching_budgets():
    groups = _build_evaluation_groups(
        ["qa_1", "niah_single_1", "qa_1", "niah_single_1", "qa_1"],
        [128, 32, 128, 32, 64],
    )

    assert groups == [
        {
            "task": "qa_1",
            "max_new_tokens": 128,
            "indices": [0, 2],
        },
        {
            "task": "niah_single_1",
            "max_new_tokens": 32,
            "indices": [1, 3],
        },
        {
            "task": "qa_1",
            "max_new_tokens": 64,
            "indices": [4],
        },
    ]


def test_evaluation_progress_bar_tracks_sample_count(monkeypatch):
    calls = []

    class FakeProgress:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
            self.n = 0

        def update(self, count):
            self.n += count
            calls.append(("update", count))

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(ruler_evaluate, "tqdm", FakeProgress, raising=False)

    progress = ruler_evaluate._make_evaluation_progress(7)
    progress.update(3)
    progress.close()

    assert calls == [
        (
            "init",
            {
                "total": 7,
                "desc": "Evaluating",
                "unit": "sample",
                "dynamic_ncols": True,
            },
        ),
        ("update", 3),
        ("close",),
    ]


def test_ruler_rejects_invalid_row_max_new_tokens():
    rows = [{"task": "qa_1", "max_new_tokens": 0}]

    try:
        _resolve_max_new_tokens(rows, override=None)
    except ValueError as error:
        assert "row 0" in str(error)
        assert "qa_1" in str(error)
    else:
        raise AssertionError("invalid RULER max_new_tokens must fail explicitly")


def test_load_rows_accepts_prompt_preserving_jsonl(tmp_path: Path):
    dataset_path = tmp_path / "ruler-32768.jsonl"
    dataset_path.write_text(
        '{"prompt":"Question?","answer_prefix":" Answer:",'
        '"answer":["42"],"task":"qa_1","max_new_tokens":32}\n',
        encoding="utf-8",
    )

    config = EvalConfig(
        model_path="/models/llama",
        output_dir=str(tmp_path / "results"),
        dataset_path=str(dataset_path),
        data_dir="32768",
    )

    rows = _load_rows(config)

    assert rows[0]["prompt"] == "Question?"
    assert rows[0]["answer_prefix"] == " Answer:"


def test_load_rows_rejects_missing_fields_in_a_later_row(tmp_path: Path):
    dataset_path = tmp_path / "ruler-32768.jsonl"
    dataset_path.write_text(
        '{"prompt":"Question 1","answer_prefix":" Answer:",'
        '"answer":["1"],"task":"qa_1","max_new_tokens":32}\n'
        '{"prompt":"Question 2","answer_prefix":" Answer:",'
        '"answer":["2"],"task":"qa_1"}\n',
        encoding="utf-8",
    )

    config = EvalConfig(
        model_path="/models/llama",
        output_dir=str(tmp_path / "results"),
        dataset_path=str(dataset_path),
    )

    with pytest.raises(ValueError, match="row 1.*max_new_tokens"):
        _load_rows(config)


def test_load_rows_downloads_length_specific_hf_file(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "downloaded.jsonl"
    dataset_path.write_text(
        '{"prompt":"Question?","answer_prefix":" Answer:",'
        '"answer":["42"],"task":"qa_1","max_new_tokens":32}\n',
        encoding="utf-8",
    )
    seen = {}

    def download(**kwargs):
        seen.update(kwargs)
        return str(dataset_path)

    monkeypatch.setattr(ruler_evaluate, "_download_hf_dataset_file", download)
    config = EvalConfig(
        model_path="/models/llama",
        output_dir=str(tmp_path / "results"),
        dataset_repo_id="org/full-ruler",
        dataset_revision="data-v1",
        data_dir="65536",
    )

    rows = _load_rows(config)

    assert len(rows) == 1
    assert seen == {
        "repo_id": "org/full-ruler",
        "filename": "ruler-65536.jsonl",
        "revision": "data-v1",
    }


def test_row_prompt_uses_preformatted_prompt_without_chat_reformatting():
    class Tokenizer:
        chat_template = "unused"

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [ord(char) for char in text]

    row = {"prompt": "RULER prompt", "answer_prefix": " Answer:"}

    assert _row_prompt(Tokenizer(), row, max_context_length=100) == [
        ord(char) for char in "RULER prompt Answer:"
    ]
