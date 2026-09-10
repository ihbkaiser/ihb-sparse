#!/usr/bin/env python3
"""Prepare and optionally upload tokenizer-aligned full RULER artifacts.

The NVIDIA RULER generator writes one ``test.jsonl`` per task. This wrapper
keeps that generator as the source of task semantics, merges the 13 official
synthetic tasks into one file per target length, and stores the preformatted
prompt so evaluation does not have to reconstruct task-specific boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
DEFAULT_LENGTHS = (32768, 65536, 131072)
TASK_MAX_NEW_TOKENS = {
    "niah": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa": 32,
}
UPSTREAM_REQUIRED_FIELDS = {"index", "input", "outputs", "answer_prefix", "length"}
ARTIFACT_REQUIRED_FIELDS = {
    "index",
    "prompt",
    "answer_prefix",
    "answer",
    "task",
    "max_new_tokens",
    "context_length",
    "seed",
    "model_template_type",
    "source_task",
    "source_index",
    "source_row_index",
    "source_length",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected an object in {path} at line {line_number}; "
                    f"got {type(value).__name__}."
                )
            rows.append(value)
    return rows


def _positive_int(value: Any, *, field: str, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source} field {field!r} must be an integer; got {value!r}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{source} field {field!r} must be an integer; got {value!r}."
        ) from error
    if parsed <= 0:
        raise ValueError(
            f"{source} field {field!r} must be positive; got {parsed}."
        )
    return parsed


def _nonnegative_int(value: Any, *, field: str, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source} field {field!r} must be an integer; got {value!r}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{source} field {field!r} must be an integer; got {value!r}."
        ) from error
    if parsed < 0:
        raise ValueError(
            f"{source} field {field!r} must be non-negative; got {parsed}."
        )
    return parsed


def convert_upstream_row(
    row: Mapping[str, Any],
    *,
    task: str,
    source_row_index: int,
    context_length: int,
    model_template_type: str,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """Convert one NVIDIA RULER row without changing its prompt semantics."""
    missing = sorted(UPSTREAM_REQUIRED_FIELDS - set(row))
    if missing:
        raise ValueError(
            f"Upstream RULER row for task {task!r} is missing fields: {missing}."
        )
    if not isinstance(row["input"], str) or not row["input"]:
        raise ValueError(f"Upstream RULER task {task!r} has an empty input prompt.")
    if not isinstance(row["answer_prefix"], str):
        raise ValueError(
            f"Upstream RULER task {task!r} has a non-string answer_prefix."
        )
    if not isinstance(model_template_type, str) or not model_template_type:
        raise ValueError("Artifact model_template_type must be a non-empty string.")
    source_index = _nonnegative_int(row["index"], field="index", source=task)
    source_row_index = _nonnegative_int(
        source_row_index, field="source_row_index", source=task
    )
    source_length = _positive_int(row["length"], field="length", source=task)
    target_length = _positive_int(
        context_length, field="context_length", source="artifact"
    )
    if source_length > target_length:
        raise ValueError(
            f"RULER task {task!r} row {source_index} length {source_length} "
            f"exceeds target context length {target_length}."
        )
    resolved_max_new_tokens = _positive_int(
        max_new_tokens, field="max_new_tokens", source=task
    )
    answers = row["outputs"]
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, (list, tuple)) or not answers:
        raise ValueError(
            f"Upstream RULER task {task!r} row {source_index} has no answers."
        )
    if not all(isinstance(answer, str) and answer for answer in answers):
        raise ValueError(
            f"Upstream RULER task {task!r} row {source_index} has invalid answers."
        )
    answers = list(answers)
    return {
        "index": source_index,
        "prompt": row["input"],
        "answer_prefix": row["answer_prefix"],
        "answer": answers,
        "task": task,
        "max_new_tokens": resolved_max_new_tokens,
        "context_length": target_length,
        "seed": int(seed),
        "model_template_type": model_template_type,
        "source_task": task,
        "source_index": source_index,
        "source_row_index": source_row_index,
        "source_length": source_length,
    }


class _ArtifactValidator:
    def __init__(
        self,
        *,
        context_length: int,
        expected_tasks: Sequence[str],
        expected_samples_per_task: int,
    ) -> None:
        self.target_length = _positive_int(
            context_length, field="context_length", source="artifact"
        )
        sample_count = _positive_int(
            expected_samples_per_task,
            field="expected_samples_per_task",
            source="artifact",
        )
        self.task_set = set(expected_tasks)
        if not self.task_set:
            raise ValueError("expected_tasks must contain at least one task.")
        self.expected_counts = {task: sample_count for task in expected_tasks}
        self.seen_source_ids: set[tuple[str, int]] = set()
        self.seen_indices: set[int] = set()
        self.counts: Counter[str] = Counter()
        self.rows_seen = 0

    def add(self, row: Mapping[str, Any], position: int) -> None:
        missing = sorted(ARTIFACT_REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(f"Artifact row {position} is missing fields: {missing}.")
        task = row["task"]
        if task not in self.task_set:
            raise ValueError(
                f"Artifact row {position} has unsupported task {task!r}; "
                f"expected one of {sorted(self.task_set)}."
            )
        if row["source_task"] != task:
            raise ValueError(
                f"Artifact row {position} has source_task={row['source_task']!r} "
                f"but task={task!r}."
            )
        if row["context_length"] != self.target_length:
            raise ValueError(
                f"Artifact row {position} has context_length={row['context_length']!r}; "
                f"expected {self.target_length}."
            )
        if not isinstance(row["prompt"], str) or not row["prompt"]:
            raise ValueError(f"Artifact row {position} has an empty prompt.")
        if not isinstance(row["answer_prefix"], str):
            raise ValueError(f"Artifact row {position} has a non-string answer_prefix.")
        if not isinstance(row["model_template_type"], str) or not row["model_template_type"]:
            raise ValueError(
                f"Artifact row {position} has an invalid model_template_type."
            )
        _positive_int(
            row["max_new_tokens"],
            field="max_new_tokens",
            source=f"row {position}",
        )
        _nonnegative_int(row["seed"], field="seed", source=f"row {position}")
        source_index = _nonnegative_int(
            row["source_index"], field="source_index", source=f"row {position}"
        )
        source_row_index = _nonnegative_int(
            row["source_row_index"],
            field="source_row_index",
            source=f"row {position}",
        )
        source_length = _positive_int(
            row["source_length"], field="source_length", source=f"row {position}"
        )
        if source_length > self.target_length:
            raise ValueError(
                f"Artifact row {position} source_length={source_length} exceeds "
                f"target context length {self.target_length}."
            )
        answers = row["answer"]
        if not isinstance(answers, (list, tuple)) or not answers:
            raise ValueError(f"Artifact row {position} has no answer references.")
        if not all(isinstance(answer, str) and answer for answer in answers):
            raise ValueError(f"Artifact row {position} has invalid answer references.")
        source_id = (str(task), source_row_index)
        if source_id in self.seen_source_ids:
            raise ValueError(f"Artifact has duplicate source identity {source_id!r}.")
        self.seen_source_ids.add(source_id)
        index = _nonnegative_int(
            row["index"], field="index", source=f"row {position}"
        )
        if index in self.seen_indices:
            raise ValueError(f"Artifact has duplicate global index {index}.")
        self.seen_indices.add(index)
        self.counts[str(task)] += 1
        self.rows_seen += 1

    def finish(self) -> None:
        if not self.rows_seen:
            raise ValueError("RULER artifact is empty.")
        if dict(self.counts) != self.expected_counts:
            raise ValueError(
                f"RULER artifact coverage mismatch: got {dict(self.counts)}, "
                f"expected {self.expected_counts}."
            )


def validate_artifact_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    context_length: int,
    expected_tasks: Sequence[str],
    expected_samples_per_task: int,
) -> None:
    """Validate complete artifact coverage and row-level schema invariants."""
    validator = _ArtifactValidator(
        context_length=context_length,
        expected_tasks=expected_tasks,
        expected_samples_per_task=expected_samples_per_task,
    )
    for position, row in enumerate(rows):
        validator.add(row, position)
    validator.finish()


def _generator_commit(ruler_repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(ruler_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _prepare_script(ruler_repo: Path) -> Path:
    script = ruler_repo / "scripts" / "data" / "prepare.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"Expected legacy NVIDIA RULER generator at {script}. "
            "Use a checkout containing scripts/data/prepare.py."
        )
    if not (ruler_repo / "scripts" / "synthetic.yaml").is_file():
        raise FileNotFoundError(
            f"NVIDIA RULER checkout is missing "
            f"{ruler_repo / 'scripts' / 'synthetic.yaml'}."
        )
    return script


def _run_upstream_task(
    *,
    ruler_repo: Path,
    work_dir: Path,
    model_path: str,
    model_template_type: str,
    task: str,
    context_length: int,
    num_samples: int,
    seed: int,
    generator_commit: str,
    python_executable: str,
) -> Path:
    script = _prepare_script(ruler_repo)
    task_dir = work_dir / str(context_length)
    output_path = task_dir / task / "test.jsonl"
    metadata_path = output_path.with_suffix(".sparsevllm-config.json")
    expected_metadata = {
        "model_path": model_path,
        "model_template_type": model_template_type,
        "task": task,
        "context_length": context_length,
        "num_samples": num_samples,
        "seed": seed,
        "generator_commit": generator_commit,
    }
    if output_path.is_file():
        if not metadata_path.is_file():
            raise RuntimeError(
                f"Refusing to reuse an untracked RULER cache at {output_path}. "
                "Use a fresh --work-dir or remove only this stale task artifact."
            )
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Cannot read RULER cache metadata {metadata_path}; refusing reuse."
            ) from error
        if cached_metadata != expected_metadata:
            raise RuntimeError(
                f"RULER cache metadata mismatch for {output_path}. "
                "Use a fresh --work-dir when changing model, template, seed, "
                "sample count, context length, or generator commit."
            )
        return output_path
    command = [
        python_executable,
        str(script),
        "--save_dir",
        str(task_dir),
        "--benchmark",
        "synthetic",
        "--task",
        task,
        "--subset",
        "test",
        "--tokenizer_path",
        model_path,
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(context_length),
        "--model_template_type",
        model_template_type,
        "--num_samples",
        str(num_samples),
        "--random_seed",
        str(seed),
    ]
    print("Running:", " ".join(command), flush=True)
    environment = os.environ.copy()
    python_dir = str(Path(python_executable).resolve().parent)
    environment["PATH"] = python_dir + os.pathsep + environment.get("PATH", "")
    subprocess.run(command, cwd=script.parent, check=True, env=environment)
    if not output_path.is_file():
        raise FileNotFoundError(
            f"NVIDIA RULER did not produce the expected artifact {output_path}."
        )
    metadata_path.write_text(
        json.dumps(expected_metadata, indent=2), encoding="utf-8"
    )
    return output_path


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _read_jsonl_for_resume(path: Path) -> list[dict[str, Any]]:
    """Read complete rows and discard an interrupted final JSONL line."""
    rows: list[dict[str, Any]] = []
    valid_bytes = 0
    interrupted_tail = False
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                valid_bytes = handle.tell()
                continue
            try:
                line = raw_line.decode("utf-8")
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if not raw_line.endswith(b"\n"):
                    interrupted_tail = True
                    break
                raise ValueError(
                    f"Invalid JSON in resumable artifact {path} at line "
                    f"{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected an object in resumable artifact {path} at line "
                    f"{line_number}; got {type(value).__name__}."
                )
            rows.append(value)
            valid_bytes = handle.tell()
    if interrupted_tail:
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
    return rows


def _artifact_metadata_path(path: Path) -> Path:
    return path.with_suffix(".sparsevllm-config.json")


def _artifact_metadata(
    *,
    model_path: str,
    model_template_type: str,
    tasks: Sequence[str],
    context_length: int,
    num_samples: int,
    seed: int,
    generator_commit: str,
) -> dict[str, Any]:
    return {
        "model_path": model_path,
        "model_template_type": model_template_type,
        "tasks": list(tasks),
        "context_length": context_length,
        "num_samples": num_samples,
        "seed": seed,
        "generator_commit": generator_commit,
    }


def _validate_cached_artifact_metadata(
    path: Path, expected_metadata: Mapping[str, Any]
) -> None:
    metadata_path = _artifact_metadata_path(path)
    if not metadata_path.is_file():
        return
    try:
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read RULER artifact cache metadata {metadata_path}; "
            "refusing reuse."
        ) from error
    if cached_metadata != dict(expected_metadata):
        raise RuntimeError(
            f"RULER artifact cache metadata mismatch for {path}. "
            "Use a fresh --output-dir when changing model, template, tasks, "
            "seed, sample count, context length, or generator commit."
        )


def _task_max_new_tokens(task: str) -> int:
    task_kind = task.split("_")[0]
    try:
        return TASK_MAX_NEW_TOKENS[task_kind]
    except KeyError as error:
        raise ValueError(
            f"No max_new_tokens mapping exists for RULER task {task!r}."
        ) from error


def _validate_artifact_prefix(
    rows: Sequence[Mapping[str, Any]],
    *,
    context_length: int,
    tasks: Sequence[str],
    num_samples: int,
    seed: int,
    model_template_type: str,
    generator_commit: str | None,
) -> None:
    """Validate an existing artifact as an ordered prefix of the target."""
    target_rows = len(tasks) * num_samples
    if len(rows) > target_rows:
        raise ValueError(
            f"RULER artifact has {len(rows)} rows; expected at most {target_rows}."
        )
    validator = _ArtifactValidator(
        context_length=context_length,
        expected_tasks=tasks,
        expected_samples_per_task=num_samples,
    )
    for position, row in enumerate(rows):
        validator.add(row, position)
        expected_task = tasks[position // num_samples]
        expected_source_row_index = position % num_samples
        if row["task"] != expected_task:
            raise ValueError(
                f"RULER artifact row {position} has task={row['task']!r}; "
                f"expected ordered task {expected_task!r}."
            )
        if row["source_row_index"] != expected_source_row_index:
            raise ValueError(
                f"RULER artifact row {position} has source_row_index="
                f"{row['source_row_index']!r}; expected {expected_source_row_index}."
            )
        if row["index"] != position:
            raise ValueError(
                f"RULER artifact row {position} has index={row['index']!r}; "
                f"expected {position}."
            )
        if row["seed"] != seed:
            raise ValueError(
                f"RULER artifact row {position} has seed={row['seed']!r}; "
                f"expected {seed}."
            )
        if row["model_template_type"] != model_template_type:
            raise ValueError(
                f"RULER artifact row {position} has model_template_type="
                f"{row['model_template_type']!r}; expected {model_template_type!r}."
            )
        expected_max_new_tokens = _task_max_new_tokens(expected_task)
        if row["max_new_tokens"] != expected_max_new_tokens:
            raise ValueError(
                f"RULER artifact row {position} has max_new_tokens="
                f"{row['max_new_tokens']!r}; expected {expected_max_new_tokens}."
            )
        if (
            generator_commit is not None
            and "generator_commit" in row
            and row["generator_commit"] != generator_commit
        ):
            raise ValueError(
                f"RULER artifact row {position} has generator_commit="
                f"{row['generator_commit']!r}; expected {generator_commit!r}."
            )
    if len(rows) == target_rows:
        validator.finish()


def _ensure_jsonl_trailing_newline(path: Path) -> None:
    if path.stat().st_size == 0:
        return
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
    with path.open("ab") as handle:
        handle.write(b"\n")


def generate_artifact(
    *,
    ruler_repo: Path,
    work_dir: Path,
    output_dir: Path,
    model_path: str,
    model_template_type: str,
    context_length: int,
    tasks: Sequence[str],
    num_samples: int,
    seed: int,
    python_executable: str,
) -> Path:
    """Generate, merge, validate, and write one context-length artifact."""
    target_length = _positive_int(
        context_length, field="context_length", source="command line"
    )
    sample_count = _positive_int(
        num_samples, field="num_samples", source="command line"
    )
    normalized_tasks = tuple(str(task) for task in tasks)
    if not normalized_tasks:
        raise ValueError("tasks must contain at least one task.")
    if len(set(normalized_tasks)) != len(normalized_tasks):
        raise ValueError("tasks must not contain duplicate task names.")
    for task in normalized_tasks:
        _task_max_new_tokens(task)
    generator_commit = _generator_commit(ruler_repo)
    output_path = output_dir / f"ruler-{target_length}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_metadata = _artifact_metadata(
        model_path=model_path,
        model_template_type=model_template_type,
        tasks=normalized_tasks,
        context_length=target_length,
        num_samples=sample_count,
        seed=seed,
        generator_commit=generator_commit,
    )
    _validate_cached_artifact_metadata(output_path, expected_metadata)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if output_path.is_file():
        if partial_path.is_file():
            raise RuntimeError(
                f"Both final and partial RULER artifacts exist: {output_path} "
                f"and {partial_path}. Remove the stale partial file explicitly."
            )
        existing_rows = _read_jsonl_for_resume(output_path)
        _validate_artifact_prefix(
            existing_rows,
            context_length=target_length,
            tasks=normalized_tasks,
            num_samples=sample_count,
            seed=seed,
            model_template_type=model_template_type,
            generator_commit=generator_commit,
        )
        if len(existing_rows) == len(normalized_tasks) * sample_count:
            print(
                f"Skipping existing complete RULER artifact: {output_path}",
                flush=True,
            )
            return output_path
        resume_path = output_path
    elif partial_path.is_file():
        _validate_cached_artifact_metadata(output_path, expected_metadata)
        existing_rows = _read_jsonl_for_resume(partial_path)
        _validate_artifact_prefix(
            existing_rows,
            context_length=target_length,
            tasks=normalized_tasks,
            num_samples=sample_count,
            seed=seed,
            model_template_type=model_template_type,
            generator_commit=generator_commit,
        )
        resume_path = partial_path
    else:
        existing_rows = []
        resume_path = partial_path
    if existing_rows:
        _ensure_jsonl_trailing_newline(resume_path)
        print(
            f"Resuming RULER artifact {output_path} after {len(existing_rows)} "
            "validated rows.",
            flush=True,
        )
    metadata_path = _artifact_metadata_path(output_path)
    if not metadata_path.is_file():
        metadata_path.write_text(
            json.dumps(expected_metadata, indent=2), encoding="utf-8"
        )

    global_index = len(existing_rows)
    with resume_path.open("a", encoding="utf-8") as output_handle:
        for task_index, task_name in enumerate(normalized_tasks):
            task_start = task_index * sample_count
            task_end = task_start + sample_count
            if len(existing_rows) >= task_end:
                global_index = task_end
                print(
                    f"Skipping existing RULER task {task_name!r} in {output_path}",
                    flush=True,
                )
                continue
            if len(existing_rows) > task_start:
                global_index = len(existing_rows)
            source_path = _run_upstream_task(
                ruler_repo=ruler_repo,
                work_dir=work_dir,
                model_path=model_path,
                model_template_type=model_template_type,
                task=task_name,
                context_length=target_length,
                num_samples=sample_count,
                seed=seed,
                generator_commit=generator_commit,
                python_executable=python_executable,
            )
            source_rows = _read_jsonl(source_path)
            if len(source_rows) != sample_count:
                raise ValueError(
                    f"NVIDIA RULER task {task_name!r} produced {len(source_rows)} rows; "
                    f"expected {sample_count}."
                )
            first_source_row_index = max(0, global_index - task_start)
            for source_row_index in range(first_source_row_index, sample_count):
                row = source_rows[source_row_index]
                artifact_row = convert_upstream_row(
                    row,
                    task=task_name,
                    source_row_index=source_row_index,
                    context_length=target_length,
                    model_template_type=model_template_type,
                    max_new_tokens=_task_max_new_tokens(task_name),
                    seed=seed,
                )
                artifact_row["index"] = global_index
                artifact_row["generator_commit"] = generator_commit
                output_handle.write(
                    json.dumps(artifact_row, ensure_ascii=False) + "\n"
                )
                global_index += 1
    completed_rows = _read_jsonl(resume_path)
    _validate_artifact_prefix(
        completed_rows,
        context_length=target_length,
        tasks=normalized_tasks,
        num_samples=sample_count,
        seed=seed,
        model_template_type=model_template_type,
        generator_commit=generator_commit,
    )
    if resume_path != output_path:
        os.replace(resume_path, output_path)
    return output_path


def upload_artifacts(
    paths: Sequence[Path],
    *,
    repo_id: str,
    revision: str,
    private: bool,
) -> None:
    """Upload only explicit artifact files to a Hugging Face dataset repo."""
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "Uploading requires huggingface_hub; install it in the active environment."
        ) from error
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    for path in paths:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            commit_message=f"Add full RULER artifact {path.name}",
        )


def _parse_csv_ints(value: str, *, field: str) -> tuple[int, ...]:
    values = tuple(
        _positive_int(item.strip(), field=field, source="command line")
        for item in value.split(",")
        if item.strip()
    )
    if not values:
        raise ValueError(f"{field} must contain at least one integer.")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicate values.")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--model-template-type",
        required=True,
        help="NVIDIA RULER template name matching the evaluated model.",
    )
    parser.add_argument("--ruler-repo", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lengths", default=",".join(map(str, DEFAULT_LENGTHS))
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-private", action="store_true")
    args = parser.parse_args()
    args.lengths = _parse_csv_ints(args.lengths, field="--lengths")
    if not args.tasks:
        raise ValueError("--tasks must contain at least one task.")
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("--tasks must not contain duplicate task names.")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    return args


def main() -> None:
    args = _parse_args()
    ruler_repo = args.ruler_repo.resolve()
    work_dir = args.work_dir.resolve()
    output_dir = args.output_dir.resolve()
    _prepare_script(ruler_repo)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = [
        generate_artifact(
            ruler_repo=ruler_repo,
            work_dir=work_dir,
            output_dir=output_dir,
            model_path=args.model_path,
            model_template_type=args.model_template_type,
            context_length=length,
            tasks=args.tasks,
            num_samples=args.num_samples,
            seed=args.seed,
            python_executable=args.python_executable,
        )
        for length in args.lengths
    ]
    manifest = {
        "model_path": args.model_path,
        "model_template_type": args.model_template_type,
        "ruler_repo": str(ruler_repo),
        "generator_commit": _generator_commit(ruler_repo),
        "lengths": list(args.lengths),
        "tasks": list(args.tasks),
        "num_samples": args.num_samples,
        "seed": args.seed,
        "files": [path.name for path in artifacts],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if args.hf_repo_id:
        upload_artifacts(
            artifacts,
            repo_id=args.hf_repo_id,
            revision=args.hf_revision,
            private=args.hf_private,
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
