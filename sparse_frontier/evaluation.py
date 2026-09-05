"""Evaluation for both legacy single-task runs and mixed RULER JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sparse_frontier.utils.data import read_jsonl


def merge_data_and_predictions(data: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge by stable sample index and fail on missing/duplicate rows."""
    data_by_index = {int(item["index"]): dict(item) for item in data}
    pred_by_index = {int(item["index"]): dict(item) for item in predictions}
    if len(data_by_index) != len(data) or len(pred_by_index) != len(predictions):
        raise ValueError("Duplicate sample indexes found in data or predictions")
    data_indexes = set(data_by_index)
    pred_indexes = set(pred_by_index)
    if data_indexes != pred_indexes:
        raise ValueError(
            "Mismatch between data and prediction indexes. "
            f"Missing predictions: {sorted(data_indexes - pred_indexes)}; "
            f"unknown predictions: {sorted(pred_indexes - data_indexes)}"
        )
    merged = []
    for index in sorted(data_indexes):
        sample = data_by_index[index]
        prediction = pred_by_index[index]
        if "task" in sample and "task" in prediction and sample["task"] != prediction["task"]:
            raise ValueError(f"Task mismatch for sample {index}: {sample['task']} vs {prediction['task']}")
        sample.update(prediction)
        merged.append(sample)
    return merged


def _task_metrics(task: str, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from sparse_frontier.ruler_kvpress import RULER_TASKS

    if task in RULER_TASKS:
        from sparse_frontier.ruler_kvpress import evaluate_ruler_task

        return evaluate_ruler_task(task, examples)
    # Preserve the pre-existing Hydra task behavior for non-pilot experiments.
    from sparse_frontier.tasks.registry import TASK_REGISTRY

    if task not in TASK_REGISTRY:
        raise ValueError(f"No evaluator registered for dataset task {task!r}")
    return TASK_REGISTRY[task].evaluate(list(examples))


def _numeric_values(examples: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for example in examples:
        value = example.get(key)
        if value is not None:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
    return values


def evaluate_jsonl_dataset(
    data_path: str | Path,
    predictions_path: str | Path,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    method: str | None = None,
    budget: int | None = None,
) -> dict[str, Any]:
    """Evaluate every task found in ``data_path``.

    There is deliberately no task argument: the task set and task membership
    are read from the dataset rows. A missing prediction is represented as a
    failed example and contributes zero, so an interrupted run cannot appear
    successful by silently shortening the denominator.
    """
    data = read_jsonl(data_path)
    predictions = read_jsonl(predictions_path) if Path(predictions_path).exists() else []
    data_by_index = {int(row["index"]): row for row in data}
    pred_by_index = {int(row["index"]): row for row in predictions}
    if len(data_by_index) != len(data) or len(pred_by_index) != len(predictions):
        raise ValueError("Duplicate sample indexes found in data or predictions")
    if set(pred_by_index) - set(data_by_index):
        raise ValueError(f"Predictions contain indexes absent from data: {sorted(set(pred_by_index) - set(data_by_index))}")

    merged: list[dict[str, Any]] = []
    for index in sorted(data_by_index):
        example = dict(data_by_index[index])
        prediction = dict(pred_by_index.get(index, {}))
        if prediction and "task" in prediction and prediction["task"] != example.get("task"):
            raise ValueError(f"Task mismatch for sample {index}: {example.get('task')} vs {prediction['task']}")
        if not prediction:
            prediction = {"pred": "", "failure": True, "error": "prediction missing"}
        example.update(prediction)
        merged.append(example)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in merged:
        task = example.get("task")
        if not task:
            raise ValueError(f"Dataset row {example.get('index')} has no task field")
        grouped[str(task)].append(example)

    task_results: list[dict[str, Any]] = []
    for task in sorted(grouped):
        examples = grouped[task]
        metrics = _task_metrics(task, examples)
        failures = sum(bool(example.get("failure") or example.get("error")) for example in examples)
        runtimes = _numeric_values(examples, "runtime_s")
        decode_latencies = _numeric_values(examples, "decode_latency_s")
        peak_memory = _numeric_values(examples, "peak_gpu_memory_bytes")
        row = {
            "method": method,
            "task": task,
            "budget": budget,
            "accuracy": float(metrics.get("accuracy", metrics.get("iou", 0.0))),
            "metric_name": "accuracy" if "accuracy" in metrics else "iou",
            "metric_variance": float(metrics.get("accuracy_variance", metrics.get("iou_variance", 0.0))),
            "total_samples": len(examples),
            "runtime_s_total": sum(runtimes) if runtimes else None,
            "runtime_s_mean": sum(runtimes) / len(runtimes) if runtimes else None,
            "decode_latency_s_total": sum(decode_latencies) if decode_latencies else None,
            "decode_latency_s_mean": sum(decode_latencies) / len(decode_latencies) if decode_latencies else None,
            "peak_gpu_memory_bytes_max": max(peak_memory) if peak_memory else None,
            "failures": failures,
            "null_predictions": int(metrics.get("null_predictions", 0)),
        }
        task_results.append(row)

    result = {
        "method": method,
        "budget": budget,
        "data_path": str(data_path),
        "predictions_path": str(predictions_path),
        "total_samples": len(merged),
        "failures": sum(row["failures"] for row in task_results),
        "tasks": task_results,
    }
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(task_results[0]) if task_results else ["method", "task", "budget", "accuracy", "failures"]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(task_results)
    return result


def evaluate_task(cfg) -> None:
    """Hydra-compatible adapter; mixed datasets are evaluated in full."""
    result = evaluate_jsonl_dataset(
        data_path=cfg.runtime.data_path,
        predictions_path=cfg.runtime.pred_path,
        output_json=cfg.runtime.results_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a mixed-task JSONL dataset")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--output_json")
    parser.add_argument("--output_csv")
    parser.add_argument("--method")
    parser.add_argument("--budget", type=int)
    args = parser.parse_args()
    result = evaluate_jsonl_dataset(
        data_path=args.data_path,
        predictions_path=args.predictions_path,
        output_json=args.output_json,
        output_csv=args.output_csv,
        method=args.method,
        budget=args.budget,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
