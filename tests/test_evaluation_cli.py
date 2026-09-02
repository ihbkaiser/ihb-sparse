import csv
import json

from sparse_frontier.evaluation import evaluate_jsonl_dataset


def _row(index, task, gold, pred, **metrics):
    row = {
        "index": index,
        "task": task,
        "input_text": "input",
        "gold_answer": gold,
        "outputs": gold if isinstance(gold, list) else [gold],
        "pred": pred,
        "output_tokens_len": 1,
    }
    row.update(metrics)
    return row


def test_mixed_dataset_evaluation_infers_tasks_and_emits_aggregates(tmp_path):
    data = tmp_path / "data.jsonl"
    pred = tmp_path / "pred.jsonl"
    out_json = tmp_path / "aggregate.json"
    out_csv = tmp_path / "aggregate.csv"

    data_rows = [
        {"index": 0, "task": "niah_single", "gold_answer": ["abc"], "outputs": ["abc"]},
        {"index": 1, "task": "vt", "gold_answer": "AAAAA BBBBB", "outputs": ["AAAAA BBBBB"]},
    ]
    pred_rows = [
        _row(0, "niah_single", ["abc"], "abc", runtime_s=1.0, decode_latency_s=0.4, peak_gpu_memory_bytes=5),
        _row(1, "vt", "AAAAA BBBBB", "AAAAA", runtime_s=2.0, decode_latency_s=0.7, peak_gpu_memory_bytes=7),
    ]
    for path, rows in ((data, data_rows), (pred, pred_rows)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    results = evaluate_jsonl_dataset(
        data_path=data,
        predictions_path=pred,
        output_json=out_json,
        output_csv=out_csv,
        method="quest",
        budget=512,
    )

    assert {row["task"] for row in results["tasks"]} == {"niah_single", "vt"}
    assert all(row["method"] == "quest" and row["budget"] == 512 for row in results["tasks"])
    assert results["failures"] == 0
    assert out_json.exists() and out_csv.exists()
    with out_csv.open(newline="") as fh:
        assert {row["task"] for row in csv.DictReader(fh)} == {"niah_single", "vt"}


def test_evaluation_reports_missing_prediction_as_failure(tmp_path):
    data = tmp_path / "data.jsonl"
    pred = tmp_path / "pred.jsonl"
    data.write_text(json.dumps({"index": 0, "task": "fwe", "gold_answer": ["aaa"]}) + "\n")
    pred.write_text("")

    results = evaluate_jsonl_dataset(data, pred)
    assert results["failures"] == 1
    assert results["tasks"][0]["failures"] == 1
    assert results["tasks"][0]["accuracy"] == 0.0
