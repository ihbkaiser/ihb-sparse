import pytest


class _BaseTokenizer:
    chat_template = None
    bos_token = "<BOS>"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]


class _ChatTokenizer:
    chat_template = "present"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking=False):
        assert add_generation_prompt is True
        assert tokenize is False
        assert enable_thinking is False
        return "<user>" + messages[0]["content"] + "</user><assistant>"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]


class _BoundaryTokenizer:
    chat_template = None
    bos_token = ""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "ab":
            return [999]
        return {"a": [1], "b\n": [2, 3]}[text]


def test_kvpress_ruler_contract_has_all_canonical_tasks_and_lengths():
    from sparse_frontier.ruler_kvpress import RULER_TASKS, SUPPORTED_CONTEXT_LENGTHS

    assert RULER_TASKS == (
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
    assert SUPPORTED_CONTEXT_LENGTHS == (4096, 8192, 16384)


def test_loader_preserves_kvpress_schema_and_validates_task_matrix():
    from sparse_frontier.ruler_kvpress import load_ruler_rows

    rows = [
        {
            "context": "document",
            "question": "Question?",
            "answer_prefix": "Answer:",
            "answer": ["correct"],
            "task": "qa_1",
            "max_new_tokens": 32,
        },
        {
            "context": "haystack",
            "question": "Needle?",
            "answer_prefix": "",
            "answer": ["needle"],
            "task": "niah_single_1",
            "max_new_tokens": 128,
        },
    ]
    calls = []

    def loader(repo_id, *, data_dir, split):
        calls.append((repo_id, data_dir, split))
        return rows

    loaded = load_ruler_rows(4096, dataset_loader=loader)

    assert calls == [("simonjegou/ruler", "4096", "test")]
    assert [row["index"] for row in loaded] == [0, 1]
    assert [row["task_index"] for row in loaded] == [0, 0]
    assert loaded[0]["gold_answer"] == ["correct"]
    assert loaded[0]["tokens_to_generate"] == 32
    assert loaded[1]["task"] == "niah_single_1"


def test_loader_rejects_invalid_context_and_unknown_task():
    from sparse_frontier.ruler_kvpress import load_ruler_rows

    with pytest.raises(ValueError, match="context_length"):
        load_ruler_rows(32768, dataset_loader=lambda *_, **__: [])
    with pytest.raises(ValueError, match="unknown task"):
        load_ruler_rows(
            4096,
            dataset_loader=lambda *_, **__: [
                {
                    "context": "x",
                    "question": "y",
                    "answer_prefix": "",
                    "answer": ["z"],
                    "task": "not_ruler",
                    "max_new_tokens": 1,
                }
            ],
        )


def test_prompt_tokens_match_kvpress_base_and_chat_preprocessing():
    from sparse_frontier.ruler_kvpress import build_prompt_token_ids

    row = {"context": "CONTEXT", "question": "QUESTION", "answer_prefix": "ANSWER:"}
    base = build_prompt_token_ids(_BaseTokenizer(), row)
    assert base == [ord(char) for char in "<BOS>CONTEXTQUESTION\nANSWER:"]

    chat = build_prompt_token_ids(_ChatTokenizer(), row)
    assert chat == [ord(char) for char in "<user>CONTEXTQUESTION</user><assistant>ANSWER:"]


def test_prompt_tokens_preserve_kvpress_context_question_token_boundary():
    from sparse_frontier.ruler_kvpress import build_prompt_token_ids

    prompt = build_prompt_token_ids(
        _BoundaryTokenizer(),
        {"context": "a", "question": "b", "answer_prefix": ""},
    )

    assert prompt == [1, 2, 3]


def test_prompt_tokens_preserve_a_preformatted_ruler_generator_input():
    from sparse_frontier.ruler_kvpress import build_prompt_token_ids

    prompt = build_prompt_token_ids(
        _ChatTokenizer(),
        {
            "prompt": "<already-rendered-ruler-input>",
            "prompt_is_preformatted": True,
            "context": "",
            "question": "",
            "answer_prefix": "",
        },
    )

    assert prompt == [ord(char) for char in "<already-rendered-ruler-input>"]


def test_scorer_matches_kvpress_all_reference_and_qa_any_reference_rules():
    from sparse_frontier.ruler_kvpress import calculate_ruler_metrics

    metrics = calculate_ruler_metrics(
        [
            {"task": "niah_multivalue", "gold_answer": ["alpha", "beta"], "pred": "alpha only"},
            {"task": "niah_multivalue", "gold_answer": ["alpha", "beta"], "pred": "alpha BETA"},
            {"task": "qa_1", "gold_answer": ["first", "accepted"], "pred": "accepted\\x00"},
            {"task": "qa_1", "gold_answer": ["first", "accepted"], "pred": "neither"},
        ]
    )

    assert metrics["niah_multivalue"]["string_match"] == 75.0
    assert metrics["qa_1"]["string_match"] == 50.0
