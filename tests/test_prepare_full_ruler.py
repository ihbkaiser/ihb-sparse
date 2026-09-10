import json
import sys
import types
from pathlib import Path

import pytest

import benchmark.kvpress_ruler.prepare_full_ruler as prepare_full_ruler
from benchmark.kvpress_ruler.prepare_full_ruler import (
    convert_upstream_row,
    generate_artifact,
    upload_artifacts,
    validate_artifact_rows,
)


def test_convert_upstream_row_preserves_prompt_and_metadata():
    row = {
        "index": 7,
        "input": "Question without the answer prefix",
        "outputs": ["42", "forty-two"],
        "answer_prefix": " Answer:",
        "length": 32720,
    }

    converted = convert_upstream_row(
        row,
        task="qa_1",
        source_row_index=0,
        context_length=32768,
        model_template_type="meta-llama3",
        max_new_tokens=32,
        seed=42,
    )

    assert converted == {
        "index": 7,
        "prompt": "Question without the answer prefix",
        "answer_prefix": " Answer:",
        "answer": ["42", "forty-two"],
        "task": "qa_1",
        "max_new_tokens": 32,
        "context_length": 32768,
        "seed": 42,
        "model_template_type": "meta-llama3",
        "source_task": "qa_1",
        "source_index": 7,
        "source_row_index": 0,
        "source_length": 32720,
    }


def test_convert_upstream_row_rejects_length_over_target():
    row = {
        "index": 0,
        "input": "prompt",
        "outputs": ["answer"],
        "answer_prefix": " Answer:",
        "length": 32769,
    }

    with pytest.raises(ValueError, match="exceeds target context length"):
        convert_upstream_row(
            row,
            task="vt",
            source_row_index=0,
            context_length=32768,
            model_template_type="meta-llama3",
            max_new_tokens=30,
            seed=42,
        )


def test_validate_artifact_rows_rejects_duplicate_task_source_identity():
    rows = [
        {
            "index": 0,
            "prompt": "p0",
            "answer_prefix": "a",
            "answer": ["x"],
            "task": "vt",
            "max_new_tokens": 30,
            "context_length": 32768,
            "seed": 42,
            "model_template_type": "meta-llama3",
            "source_index": 1,
            "source_row_index": 0,
            "source_task": "vt",
            "source_length": 32760,
        },
        {
            "index": 1,
            "prompt": "p1",
            "answer_prefix": "a",
            "answer": ["y"],
            "task": "vt",
            "max_new_tokens": 30,
            "context_length": 32768,
            "seed": 42,
            "model_template_type": "meta-llama3",
            "source_index": 1,
            "source_row_index": 0,
            "source_task": "vt",
            "source_length": 32760,
        },
    ]

    with pytest.raises(ValueError, match="duplicate source identity"):
        validate_artifact_rows(
            rows,
            context_length=32768,
            expected_tasks=("vt",),
            expected_samples_per_task=2,
        )


def test_validate_artifact_rows_allows_repeated_upstream_index_by_row_ordinal():
    rows = []
    for index in range(2):
        rows.append(
            {
                "index": index,
                "prompt": f"p{index}",
                "answer_prefix": "a",
                "answer": ["x"],
                "task": "niah_single_1",
                "max_new_tokens": 128,
                "context_length": 32768,
                "seed": 42,
                "model_template_type": "meta-llama3",
                "source_index": 7,
                "source_row_index": index,
                "source_task": "niah_single_1",
                "source_length": 32760,
            }
        )

    validate_artifact_rows(
        rows,
        context_length=32768,
        expected_tasks=("niah_single_1",),
        expected_samples_per_task=2,
    )


def test_upload_artifacts_targets_only_the_three_jsonl_files(tmp_path, monkeypatch):
    paths = []
    for name in ("ruler-32768.jsonl", "ruler-65536.jsonl", "ruler-131072.jsonl"):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        paths.append(path)

    class FakeApi:
        created = []
        uploaded = []

        def create_repo(self, **kwargs):
            self.created.append(kwargs)

        def upload_file(self, **kwargs):
            self.uploaded.append(kwargs)

    fake_api = FakeApi()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=lambda: fake_api),
    )

    upload_artifacts(
        paths,
        repo_id="org/full-ruler",
        revision="main",
        private=True,
    )

    assert fake_api.created == [
        {
            "repo_id": "org/full-ruler",
            "repo_type": "dataset",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert [item["path_in_repo"] for item in fake_api.uploaded] == [
        "ruler-32768.jsonl",
        "ruler-65536.jsonl",
        "ruler-131072.jsonl",
    ]
    assert all(item["revision"] == "main" for item in fake_api.uploaded)


def test_generate_artifact_streams_tasks_into_one_validated_jsonl(tmp_path, monkeypatch):
    def fake_generator_commit(_ruler_repo):
        return "ruler-commit"

    def fake_run_upstream_task(**kwargs):
        path = (
            kwargs["work_dir"]
            / str(kwargs["context_length"])
            / kwargs["task"]
            / "test.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "index": 7,
                "input": f"prompt {index}",
                "outputs": [f"answer {index}"],
                "answer_prefix": " Answer:",
                "length": 32760,
            }
            for index in range(kwargs["num_samples"])
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    monkeypatch.setattr(prepare_full_ruler, "_generator_commit", fake_generator_commit)
    monkeypatch.setattr(
        prepare_full_ruler, "_run_upstream_task", fake_run_upstream_task
    )

    output_path = generate_artifact(
        ruler_repo=tmp_path / "ruler",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
        model_path="/models/llama",
        model_template_type="meta-llama3",
        context_length=32768,
        tasks=("niah_single_1", "vt"),
        num_samples=2,
        seed=42,
        python_executable="python",
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 4
    assert [row["index"] for row in rows] == [0, 1, 2, 3]
    assert [row["source_row_index"] for row in rows] == [0, 1, 0, 1]
    assert all(row["generator_commit"] == "ruler-commit" for row in rows)


def test_generate_artifact_skips_complete_existing_artifact(
    tmp_path, monkeypatch
):
    def fake_generator_commit(_ruler_repo):
        return "ruler-commit"

    calls = []

    def fake_run_upstream_task(**kwargs):
        calls.append(kwargs["task"])
        path = (
            kwargs["work_dir"]
            / str(kwargs["context_length"])
            / kwargs["task"]
            / "test.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "index": index,
                "input": f"prompt {kwargs['task']} {index}",
                "outputs": [f"answer {index}"],
                "answer_prefix": " Answer:",
                "length": 32760,
            }
            for index in range(kwargs["num_samples"])
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    monkeypatch.setattr(prepare_full_ruler, "_generator_commit", fake_generator_commit)
    monkeypatch.setattr(
        prepare_full_ruler, "_run_upstream_task", fake_run_upstream_task
    )

    generate_kwargs = {
        "ruler_repo": tmp_path / "ruler",
        "work_dir": tmp_path / "work",
        "output_dir": tmp_path / "out",
        "model_path": "/models/llama",
        "model_template_type": "meta-llama3",
        "context_length": 32768,
        "tasks": ("niah_single_1", "vt"),
        "num_samples": 2,
        "seed": 42,
        "python_executable": "python",
    }
    output_path = generate_artifact(**generate_kwargs)
    assert calls == ["niah_single_1", "vt"]
    original_content = output_path.read_text(encoding="utf-8")

    def fail_if_called(**_kwargs):
        raise AssertionError("complete artifact must not invoke the generator")

    monkeypatch.setattr(prepare_full_ruler, "_run_upstream_task", fail_if_called)
    assert generate_artifact(**generate_kwargs) == output_path
    assert output_path.read_text(encoding="utf-8") == original_content


def test_generate_artifact_resumes_partial_without_reprocessing_completed_tasks(
    tmp_path, monkeypatch
):
    def fake_generator_commit(_ruler_repo):
        return "ruler-commit"

    partial_path = tmp_path / "out" / "ruler-32768.jsonl.partial"
    partial_path.parent.mkdir(parents=True)
    existing_rows = []
    for source_row_index in range(2):
        row = prepare_full_ruler.convert_upstream_row(
            {
                "index": 7,
                "input": f"prompt niah_single_1 {source_row_index}",
                "outputs": [f"answer {source_row_index}"],
                "answer_prefix": " Answer:",
                "length": 32760,
            },
            task="niah_single_1",
            source_row_index=source_row_index,
            context_length=32768,
            model_template_type="meta-llama3",
            max_new_tokens=128,
            seed=42,
        )
        row["index"] = source_row_index
        row["generator_commit"] = "ruler-commit"
        existing_rows.append(row)
    partial_path.write_text(
        "".join(json.dumps(row) + "\n" for row in existing_rows), encoding="utf-8"
    )

    calls = []

    def fake_run_upstream_task(**kwargs):
        calls.append(kwargs["task"])
        path = (
            kwargs["work_dir"]
            / str(kwargs["context_length"])
            / kwargs["task"]
            / "test.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "index": 7,
                "input": f"prompt {kwargs['task']} {index}",
                "outputs": [f"answer {index}"],
                "answer_prefix": " Answer:",
                "length": 32760,
            }
            for index in range(kwargs["num_samples"])
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    monkeypatch.setattr(prepare_full_ruler, "_generator_commit", fake_generator_commit)
    monkeypatch.setattr(
        prepare_full_ruler, "_run_upstream_task", fake_run_upstream_task
    )

    output_path = generate_artifact(
        ruler_repo=tmp_path / "ruler",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
        model_path="/models/llama",
        model_template_type="meta-llama3",
        context_length=32768,
        tasks=("niah_single_1", "vt"),
        num_samples=2,
        seed=42,
        python_executable="python",
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert calls == ["vt"]
    assert len(rows) == 4
    assert [row["task"] for row in rows] == [
        "niah_single_1",
        "niah_single_1",
        "vt",
        "vt",
    ]


def test_upstream_cache_mismatch_fails_before_reuse(tmp_path):
    ruler_repo = tmp_path / "ruler"
    (ruler_repo / "scripts" / "data").mkdir(parents=True)
    (ruler_repo / "scripts" / "data" / "prepare.py").write_text("", encoding="utf-8")
    (ruler_repo / "scripts" / "synthetic.yaml").write_text("", encoding="utf-8")
    output_path = tmp_path / "work" / "32768" / "vt" / "test.jsonl"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("{}\n", encoding="utf-8")
    output_path.with_suffix(".sparsevllm-config.json").write_text(
        json.dumps({"seed": 1}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="cache metadata mismatch"):
        prepare_full_ruler._run_upstream_task(
            ruler_repo=ruler_repo,
            work_dir=tmp_path / "work",
            model_path="/models/llama",
            model_template_type="meta-llama3",
            task="vt",
            context_length=32768,
            num_samples=2,
            seed=42,
            generator_commit="ruler-commit",
            python_executable="python",
        )
