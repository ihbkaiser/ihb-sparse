"""KVPress-compatible RULER dataset, prompt, and metric adapters.

The canonical data source and metric behavior are taken from NVIDIA KVPress's
``evaluation/benchmarks/ruler`` implementation.  Sparse inference remains in
this repository's vLLM backend; this module deliberately contains no vLLM
dependency so that dataset and scoring contracts stay easy to test.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence


RULER_DATASET = "simonjegou/ruler"
SUPPORTED_CONTEXT_LENGTHS = (4096, 8192, 16384)
RULER_TASKS = (
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

_REQUIRED_FIELDS = ("context", "question", "answer_prefix", "answer", "task", "max_new_tokens")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f]")


def _as_rows(dataset: Any) -> list[Mapping[str, Any]]:
    if hasattr(dataset, "to_list"):
        values = dataset.to_list()
    else:
        values = list(dataset)
    if not all(isinstance(value, Mapping) for value in values):
        raise ValueError("RULER dataset rows must be mappings")
    return values


def _answers(value: Any) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("RULER answer must be a non-string sequence of reference strings")
    result = [str(item) for item in value]
    if not result:
        raise ValueError("RULER answer must contain at least one reference")
    return result


def load_ruler_rows(
    context_length: int,
    *,
    dataset_loader: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Load one canonical KVPress RULER configuration into stable run rows."""
    if int(context_length) not in SUPPORTED_CONTEXT_LENGTHS:
        raise ValueError(
            f"context_length must be one of {SUPPORTED_CONTEXT_LENGTHS}, got {context_length!r}"
        )
    if dataset_loader is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - exercised in installation, not unit tests
            raise RuntimeError("RULER evaluation requires `datasets`; install requirements.txt") from exc
        dataset_loader = load_dataset

    # KVPress passes the context configuration as ``data_dir`` rather than a
    # dataset-builder configuration.  The distinction matters for the Hub's
    # plain Parquet repository layout.
    source_rows = _as_rows(
        dataset_loader(RULER_DATASET, data_dir=str(context_length), split="test")
    )
    if not source_rows:
        raise ValueError(f"KVPress RULER configuration {context_length} is empty")

    rows: list[dict[str, Any]] = []
    task_indexes: dict[str, int] = defaultdict(int)
    for index, source in enumerate(source_rows):
        missing = [field for field in _REQUIRED_FIELDS if field not in source]
        if missing:
            raise ValueError(f"RULER row {index} is missing required fields: {', '.join(missing)}")
        task = str(source["task"])
        if task not in RULER_TASKS:
            raise ValueError(f"RULER row {index} has unknown task {task!r}")
        max_new_tokens = int(source["max_new_tokens"])
        if max_new_tokens < 1:
            raise ValueError(f"RULER row {index} has invalid max_new_tokens={max_new_tokens}")
        answers = _answers(source["answer"])
        rows.append(
            {
                "index": index,
                "task": task,
                "task_index": task_indexes[task],
                "context": str(source["context"]),
                "question": str(source["question"]),
                "answer_prefix": str(source["answer_prefix"]),
                "answer": answers,
                "gold_answer": answers,
                "max_new_tokens": max_new_tokens,
                "tokens_to_generate": max_new_tokens,
                "context_length": int(context_length),
            }
        )
        task_indexes[task] += 1
    return rows


def build_prompt_token_ids(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    enable_thinking: bool = False,
) -> list[int]:
    """Mirror ``KVPressTextGenerationPipeline.preprocess`` for one RULER row."""
    if bool(row.get("prompt_is_preformatted", False)):
        if "prompt" not in row:
            raise ValueError("preformatted RULER row is missing prompt")
        # RULER's generator may already serialize model-specific role markers.
        # Calling apply_chat_template here would make methods attend to a
        # different prompt than the pinned JSONL bytes.
        return list(tokenizer.encode(str(row["prompt"]), add_special_tokens=False))
    context = str(row["context"])
    question = str(row["question"])
    answer_prefix = str(row["answer_prefix"])
    if getattr(tokenizer, "chat_template", None) is None:
        context_prefix = str(getattr(tokenizer, "bos_token", "") or "") + context
        question_suffix = "\n"
    else:
        separator = "#" * (len(context) + 10)
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": context + separator}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        try:
            context_prefix, question_suffix = formatted.split(separator)
        except ValueError as exc:
            raise ValueError("chat template did not preserve the KVPress context separator") from exc
    # KVPress tokenizes the compressed context and question suffix separately;
    # joining their strings first can change BPE merges at the boundary.
    context_ids = tokenizer.encode(context_prefix, add_special_tokens=False)
    question_ids = tokenizer.encode(
        question + question_suffix + answer_prefix, add_special_tokens=False
    )
    return list(context_ids) + list(question_ids)


def _normalize_prediction(value: Any) -> str:
    return _CONTROL_CHARS.sub("", str(value).strip()).strip()


def _string_match_all(predictions: Iterable[str], references: Iterable[Sequence[str]]) -> float:
    pairs = list(zip(predictions, references))
    if not pairs:
        raise ValueError("cannot score an empty RULER task")
    score = sum(
        sum(1.0 if ref.lower() in pred.lower() else 0.0 for ref in refs) / len(refs)
        for pred, refs in pairs
    ) / len(pairs) * 100
    return round(score, 2)


def _string_match_part(predictions: Iterable[str], references: Iterable[Sequence[str]]) -> float:
    pairs = list(zip(predictions, references))
    if not pairs:
        raise ValueError("cannot score an empty RULER task")
    score = sum(
        max(1.0 if ref.lower() in pred.lower() else 0.0 for ref in refs)
        for pred, refs in pairs
    ) / len(pairs) * 100
    return round(score, 2)


def calculate_ruler_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Return KVPress's per-task RULER string-match metrics (0--100)."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row["task"])
        if task not in RULER_TASKS:
            raise ValueError(f"unknown RULER task {task!r}")
        grouped[task].append(row)

    metrics: dict[str, dict[str, float]] = {}
    for task, task_rows in sorted(grouped.items()):
        predictions = [_normalize_prediction(row.get("pred", row.get("predicted_answer", ""))) for row in task_rows]
        references = [_answers(row.get("gold_answer", row.get("answer"))) for row in task_rows]
        metric = _string_match_part if task.split("_", 1)[0] == "qa" else _string_match_all
        metrics[task] = {"string_match": metric(predictions, references)}
    return metrics


def evaluate_ruler_task(task: str, rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Adapter for the repository-wide evaluator, expressed as 0--1 accuracy."""
    scores = calculate_ruler_metrics(rows)
    if task not in scores:
        raise ValueError(f"no examples supplied for RULER task {task!r}")
    score = scores[task]["string_match"]
    return {"accuracy": score / 100.0, "string_match": score}
