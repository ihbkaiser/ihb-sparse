import sys

from benchmark.kvpress_ruler.evaluate import (
    EvalConfig,
    _build_infer_config,
    _parse_args,
    _resolve_max_new_tokens,
)


def test_kvpress_defaults_are_paper_aligned_for_sparse_comparisons(monkeypatch):
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
    assert config.shadowkv_storage == "cpu"


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


def test_ruler_rejects_invalid_row_max_new_tokens():
    rows = [{"task": "qa_1", "max_new_tokens": 0}]

    try:
        _resolve_max_new_tokens(rows, override=None)
    except ValueError as error:
        assert "row 0" in str(error)
        assert "qa_1" in str(error)
    else:
        raise AssertionError("invalid RULER max_new_tokens must fail explicitly")
