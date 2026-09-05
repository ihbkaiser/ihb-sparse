import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from sparse_frontier.modelling.attention.efficient_decoding import QuestAttention


def test_quest_native_recipe_keeps_exactly_the_first_two_decode_layers_dense():
    attention = QuestAttention(
        token_budget=2048,
        page_size=16,
        max_input_tokens=131072,
        max_output_tokens=128,
        num_layers=32,
        share_pages=True,
        dense_layers=2,
    )

    assert attention.uses_dense_decode(0) is True
    assert attention.uses_dense_decode(1) is True
    assert attention.uses_dense_decode(2) is False
