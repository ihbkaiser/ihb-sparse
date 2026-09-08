from benchmark.kvpress_ruler.evaluate import _resolve_max_new_tokens


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
