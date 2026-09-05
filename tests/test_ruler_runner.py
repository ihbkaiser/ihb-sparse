from sparse_frontier.ruler_runner import (
    RunSpec,
    _capture_context_payload,
    _configure_attention,
    _load_model_config,
    _select_tasks,
    _select_task_indices,
    _generate_one,
    build_run_matrix,
    validate_budget,
)


def test_run_matrix_is_exactly_requested():
    matrix = build_run_matrix()
    assert [(x.method, x.budget) for x in matrix] == [
        ("dense", None),
        ("quest", 96),
        ("quest", 128),
        ("quest", 256),
        ("quest", 512),
        ("quest", 1024),
        ("quest", 2048),
        ("shadowkv", 96),
        ("shadowkv", 128),
        ("shadowkv", 256),
        ("shadowkv", 512),
        ("shadowkv", 1024),
        ("shadowkv", 2048),
    ]


def test_smoke_matrix_has_one_run_per_method():
    matrix = build_run_matrix(smoke=True)
    assert [(x.method, x.budget) for x in matrix] == [
        ("dense", None),
        ("quest", 512),
        ("shadowkv", 512),
    ]


def test_budget_validation():
    validate_budget("dense", None)
    validate_budget("query_robust", 96)
    validate_budget("query_robust", 128)
    validate_budget("query_robust", 256)
    validate_budget("quest", 512)
    validate_budget("shadowkv", 2048)
    validate_budget("query_robust", 1024)

    import pytest

    with pytest.raises(ValueError):
        validate_budget("dense", 512)
    with pytest.raises(ValueError):
        validate_budget("quest", 513)
    with pytest.raises(ValueError):
        validate_budget("unsupported", 512)


def test_shadowkv_runner_defaults_to_randomized_svd(monkeypatch):
    monkeypatch.delenv("SF_ATTENTION_ARGS_JSON", raising=False)
    _configure_attention(
        RunSpec("shadowkv", 512),
        {
            "tp": 1,
            "num_q_heads": 32,
            "num_kv_heads": 8,
            "num_layers": 32,
            "max_input_tokens": 8192,
            "max_output_tokens": 1024,
            "kv_cache_block_size": 256,
        },
    )
    import json

    args = json.loads(__import__("os").environ["SF_ATTENTION_ARGS_JSON"])
    assert args["svd_backend"] == "randomized"
    assert args["fused_retrieval"] is True


def test_dense_capture_enables_patch_and_model_geometry(monkeypatch, tmp_path):
    monkeypatch.delenv("SF_QUERY_CAPTURE_DIR", raising=False)
    _configure_attention(
        RunSpec("dense", None),
        {
            "tp": 1,
            "num_q_heads": 32,
            "num_kv_heads": 8,
            "num_layers": 32,
            "head_dim": 128,
            "max_input_tokens": 8192,
            "max_output_tokens": 4,
            "kv_cache_block_size": 256,
        },
        capture_query_dir=tmp_path / "capture",
        capture_context_path=tmp_path / "context.json",
        capture_layers=(0, 7, 31),
    )
    import json
    import os

    assert os.environ["SF_USE_ATTENTION_PATCH"] == "1"
    assert "SF_ATTENTION_NAME" not in os.environ
    assert os.environ["SF_MODEL_HEAD_DIM"] == "128"
    assert json.loads(os.environ["SF_QUERY_CAPTURE_LAYERS"]) == [0, 7, 31]


def test_dense_capture_prompt_query_options_are_explicit_and_manifested(monkeypatch, tmp_path):
    import json
    import os

    _configure_attention(
        RunSpec("dense", None),
        {
            "tp": 1,
            "num_q_heads": 32,
            "num_kv_heads": 8,
            "num_layers": 32,
            "head_dim": 128,
            "max_input_tokens": 16384,
            "max_output_tokens": 128,
            "kv_cache_block_size": 256,
        },
        capture_query_dir=tmp_path / "capture",
        capture_context_path=tmp_path / "context.json",
        capture_prompt_keys=False,
        capture_prompt_queries=True,
        prompt_query_samples=16,
        prompt_query_seed=1043,
    )
    assert os.environ["SF_QUERY_CAPTURE_PROMPT_QUERIES"] == "1"
    assert os.environ["SF_QUERY_CAPTURE_PROMPT_QUERY_SAMPLES"] == "16"
    assert os.environ["SF_QUERY_CAPTURE_PROMPT_QUERY_SEED"] == "1043"


def test_query_robust_can_be_configured_with_request_prompt_root(monkeypatch, tmp_path):
    import json
    import os

    monkeypatch.delenv("SF_QUERY_ROBUST_PROMPT_QUERY_ROOT", raising=False)
    pool = tmp_path / "pool"
    pool.mkdir()
    model_cfg = {
        "tp": 1,
        "num_q_heads": 32,
        "num_kv_heads": 8,
        "num_layers": 32,
        "head_dim": 128,
        "max_input_tokens": 16384,
        "max_output_tokens": 128,
        "kv_cache_block_size": 256,
        "model_id": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "model_revision": "d" * 40,
        "rope_type": "llama3",
        "rope_parameters": {"factor": 8.0, "rope_theta": 500000.0},
        "attention_scale": 128**-0.5,
    }
    request_root = tmp_path / "prompt_sources"
    _configure_attention(
        RunSpec("query_robust", 1024),
        model_cfg,
        query_pool_path=pool,
        query_robust_prompt_query_root=request_root,
        query_robust_generation_horizon=128,
    )
    args = json.loads(os.environ["SF_ATTENTION_ARGS_JSON"])
    assert args["prompt_query_root"] == str(request_root.resolve())
    assert os.environ["SF_QUERY_ROBUST_PROMPT_QUERY_ROOT"] == str(request_root.resolve())


def test_query_robust_prompt_source_requires_tp1_and_overrides_stale_worker_mode(
    monkeypatch, tmp_path
):
    import os

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    with __import__("pytest").raises(ValueError, match="TP=1"):
        _configure_attention(
            RunSpec("query_robust", 1024),
            {
                "tp": 2,
                "num_q_heads": 32,
                "num_kv_heads": 8,
                "num_layers": 32,
                "head_dim": 128,
                "max_input_tokens": 16384,
                "max_output_tokens": 128,
                "kv_cache_block_size": 256,
                "model_id": "test/model",
                "model_revision": "d" * 40,
                "rope_type": "llama3",
                "rope_parameters": {"factor": 8.0, "rope_theta": 500000.0},
                "attention_scale": 128**-0.5,
            },
            query_pool_path=tmp_path / "pool",
            query_robust_prompt_query_root=tmp_path / "prompt",
        )
    _configure_attention(
        RunSpec("query_robust", 1024),
        {
            "tp": 1,
            "num_q_heads": 32,
            "num_kv_heads": 8,
            "num_layers": 32,
            "head_dim": 128,
            "max_input_tokens": 16384,
            "max_output_tokens": 128,
            "kv_cache_block_size": 256,
            "model_id": "test/model",
            "model_revision": "d" * 40,
            "rope_type": "llama3",
            "rope_parameters": {"factor": 8.0, "rope_theta": 500000.0},
            "attention_scale": 128**-0.5,
        },
        query_pool_path=tmp_path / "pool",
    )
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"


def test_dense_configuration_clears_stale_prompt_query_root(monkeypatch, tmp_path):
    import os

    monkeypatch.setenv("SF_QUERY_ROBUST_PROMPT_QUERY_ROOT", str(tmp_path / "stale"))
    _configure_attention(
        RunSpec("dense", None),
        {
            "tp": 1,
            "num_q_heads": 32,
            "num_kv_heads": 8,
            "num_layers": 32,
            "head_dim": 128,
            "max_input_tokens": 8192,
            "max_output_tokens": 128,
            "kv_cache_block_size": 256,
        },
    )
    assert "SF_QUERY_ROBUST_PROMPT_QUERY_ROOT" not in os.environ


def test_capture_context_has_no_prompt_text():
    payload = _capture_context_payload(
        {
            "index": 6,
            "task": "niah_multikey_1",
            "task_index": 6,
            "context_length": 8192,
            "context": "secret context",
            "question": "secret question",
        },
        prompt_length=8181,
    )
    assert payload["split"] == "validation"
    assert payload["prompt_length"] == 8181
    assert "secret context" not in repr(payload)
    assert "secret question" not in repr(payload)
    assert 0 <= payload["sequence_id_hash"] < 2**63


def test_model_geometry_is_loaded_from_checkpoint(tmp_path):
    import json

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "hidden_size": 2048,
            }
        ),
        encoding="utf-8",
    )
    config = _load_model_config(tmp_path, tp=2, max_input_tokens=4096, max_output_tokens=8)
    assert config["num_layers"] == 24
    assert config["num_q_heads"] == 16
    assert config["num_kv_heads"] == 4
    assert config["head_dim"] == 128


def test_select_task_indices_keeps_each_task_and_requested_split():
    rows = [
        {"task": task, "task_index": index}
        for task in ("niah_single_1", "vt")
        for index in range(7)
    ]
    selected = _select_task_indices(rows, (0, 5))
    assert [(row["task"], row["task_index"]) for row in selected] == [
        ("niah_single_1", 0),
        ("niah_single_1", 5),
        ("vt", 0),
        ("vt", 5),
    ]


def test_select_tasks_keeps_exact_requested_families():
    rows = [
        {"task": task, "task_index": index}
        for task in ("niah_single_1", "fwe", "vt")
        for index in range(2)
    ]
    selected = _select_tasks(rows, ("fwe", "niah_single_1"))
    assert [(row["task"], row["task_index"]) for row in selected] == [
        ("niah_single_1", 0),
        ("niah_single_1", 1),
        ("fwe", 0),
        ("fwe", 1),
    ]


def test_query_robust_runner_sets_fail_closed_identity(monkeypatch, tmp_path):
    import json
    import os

    pool = tmp_path / "pool"
    pool.mkdir()
    model_cfg = {
        "tp": 1,
        "num_q_heads": 32,
        "num_kv_heads": 8,
        "num_layers": 32,
        "head_dim": 128,
        "max_input_tokens": 8192,
        "max_output_tokens": 8,
        "kv_cache_block_size": 256,
        "model_id": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "model_revision": "d" * 40,
        "rope_type": "llama3",
        "rope_parameters": {"factor": 8.0, "rope_theta": 500000.0},
        "attention_scale": 128**-0.5,
    }
    _configure_attention(
        RunSpec("query_robust", 1024),
        model_cfg,
        query_pool_path=pool,
        query_robust_generation_horizon=8,
        query_robust_recent_chunks=3,
    )
    args = json.loads(os.environ["SF_ATTENTION_ARGS_JSON"])
    assert args["chunk_size"] == 16
    assert args["generation_horizon"] == 8
    assert args["recent_chunks"] == 3
    assert args["query_pool_path"] == str(pool.resolve())
    assert args["bias_mode"] == "raw_entropy"
    assert args["share_chunks_across_kv_heads"] is True
    assert os.environ["SF_MODEL_REVISION"] == "d" * 40


def test_generate_one_never_exceeds_model_declared_output_horizon():
    class Model:
        max_output_tokens = 8
        enable_thinking = False

        def __init__(self):
            self.requested = None
            raw_tokenizer = type(
                "RawTokenizer",
                (),
                {
                    "chat_template": None,
                    "bos_token": "",
                    "encode": staticmethod(lambda text, add_special_tokens=False: [1, 2, 3]),
                },
            )()
            self.tokenizer = type("Tokenizer", (), {"tokenizer": raw_tokenizer})()

        def generate_token_ids(self, token_ids, max_tokens):
            self.requested = (token_ids, max_tokens)
            return {
                "text": "ok",
                "output_tokens_len": max_tokens,
                "runtime_s": 0.0,
            }

    model = Model()
    result = _generate_one(
        model,
        {
            "index": 0,
            "task": "niah_single_1",
            "context": "context",
            "question": "question",
            "answer_prefix": "",
            "tokens_to_generate": 128,
        },
        RunSpec("dense", None),
    )
    assert not result["failure"]
    assert model.requested == ([1, 2, 3, 1, 2, 3], 8)
