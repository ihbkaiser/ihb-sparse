import json

import pytest

from sparse_frontier.ruler_pilot import (
    PILOT_TASKS,
    build_pilot_rows,
    load_pilot_rows,
    write_pilot_once,
)


class FakeTokenizer:
    """Small deterministic tokenizer for generator/schema tests."""

    def text_to_tokens(self, text):
        return text.split()

    def encode_for_generation(self, text, return_tensors=False):
        return {"input_ids": self.text_to_tokens(text)}


ESSAY = "A short essay sentence. Another useful sentence. " * 20


def test_pilot_task_set_and_exact_counts():
    rows = build_pilot_rows(
        tokenizer=FakeTokenizer(),
        samples_per_task=2,
        max_seq_length=256,
        tokens_to_generate=8,
        seed=17,
        essay_text=ESSAY,
    )

    assert tuple(PILOT_TASKS) == (
        "niah_single",
        "niah_multikey",
        "niah_multiquery",
        "vt",
        "fwe",
    )
    assert len(rows) == 10
    assert [row["index"] for row in rows] == list(range(10))
    assert {row["task"] for row in rows} == set(PILOT_TASKS)
    assert all(row["input_text"] and row["gold_answer"] for row in rows)
    assert all(row["context_length"] == 256 for row in rows)
    assert all("task_args" in row and "ruler_task" in row for row in rows)


def test_pilot_generation_is_deterministic():
    kwargs = dict(
        tokenizer=FakeTokenizer(),
        samples_per_task=1,
        max_seq_length=256,
        tokens_to_generate=8,
        seed=123,
        essay_text=ESSAY,
    )
    assert build_pilot_rows(**kwargs) == build_pilot_rows(**kwargs)


def test_write_pilot_once_refuses_overwrite(tmp_path):
    output = tmp_path / "pilot.jsonl"
    rows = build_pilot_rows(
        tokenizer=FakeTokenizer(),
        samples_per_task=1,
        max_seq_length=256,
        tokens_to_generate=8,
        seed=1,
        essay_text=ESSAY,
    )
    write_pilot_once(output, rows)
    with pytest.raises(FileExistsError):
        write_pilot_once(output, rows)
    assert load_pilot_rows(output) == rows
    assert len(output.read_text().splitlines()) == 5


def test_pilot_rows_are_jsonl_serializable(tmp_path):
    rows = build_pilot_rows(
        tokenizer=FakeTokenizer(),
        samples_per_task=1,
        max_seq_length=256,
        tokens_to_generate=8,
        seed=1,
        essay_text=ESSAY,
    )
    output = tmp_path / "pilot.jsonl"
    write_pilot_once(output, rows)
    for line in output.read_text().splitlines():
        assert isinstance(json.loads(line), dict)
