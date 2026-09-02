import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from sparse_frontier.modelling.attention.handler import AttentionHandler
from sparse_frontier.modelling.attention.registry import ATTENTION_REGISTRY
from sparse_frontier.modelling.attention.abstract_attention import AttentionUtils


def test_shadowkv_is_registered_with_randomized_default():
    assert "shadowkv" in ATTENTION_REGISTRY
    config = Path("sparse_frontier/configs/attention/shadowkv.yaml").read_text()
    for value in ("sparse_budget: 2048", "chunk_size: 8", "rank: 160", "local_chunk: 4", "outlier_chunk: 48"):
        assert value in config
    assert "svd_backend: randomized" in config


def test_vllm_cache_is_reordered_to_logical_head_major_view():
    # vLLM FlashAttention stores [block, token, head, dim], while handlers
    # gather [head, logical_token, dim].  This catches the tempting but wrong
    # plain-view implementation.
    physical = torch.arange(2 * 4 * 3 * 2, dtype=torch.float32).view(2, 4, 3, 2)
    cache = torch.stack([physical, physical + 1000])
    key, value = AttentionUtils.reshape_kv_cache(cache, target_block_size=4, max_blocks=2)
    torch.testing.assert_close(key, physical.permute(2, 0, 1, 3))
    torch.testing.assert_close(value, (physical + 1000).permute(2, 0, 1, 3))

    remapped = torch.tensor([[1, 0]], dtype=torch.int32)
    key, value = AttentionUtils.reshape_kv_cache(
        cache, target_block_size=4, max_blocks=2, block_table=remapped
    )
    torch.testing.assert_close(key, physical[[1, 0]].permute(2, 0, 1, 3))
    torch.testing.assert_close(value, (physical + 1000)[[1, 0]].permute(2, 0, 1, 3))


def test_handler_captures_a_clone_and_consumes_it_once(monkeypatch):
    handler = AttentionHandler(
        tp_size=1,
        model_q_heads=4,
        model_kv_heads=2,
        model_layers=1,
        max_input_tokens=64,
        max_output_tokens=8,
        block_size=16,
    )
    key = torch.randn(3, 2, 4)
    positions = torch.arange(3)
    rope_forward = object()
    handler.capture_pre_rope(key, positions, rope_forward)
    key.add_(100)
    captured_key, captured_positions, captured_rope = handler.consume_pre_rope()
    assert torch.all(captured_key < 100)
    torch.testing.assert_close(captured_positions, positions)
    assert captured_rope is rope_forward
    with pytest.raises(RuntimeError, match="pre-RoPE"):
        handler.consume_pre_rope()


def test_handler_rejects_non_scalar_rope_positions():
    handler = AttentionHandler(
        tp_size=1,
        model_q_heads=4,
        model_kv_heads=2,
        model_layers=1,
        max_input_tokens=64,
        max_output_tokens=8,
        block_size=16,
    )
    with pytest.raises(RuntimeError, match="RoPE|position"):
        handler.capture_pre_rope(torch.randn(3, 8), torch.zeros(2, 3, dtype=torch.long), object())


def test_vllm_rope_hook_captures_before_in_place_mutation(monkeypatch):
    from sparse_frontier.modelling.models import vllm_model

    monkeypatch.setenv("SF_ATTENTION_NAME", "shadowkv")

    class FakeHandler:
        def __init__(self):
            self.capture = None

        def capture_pre_rope(self, key, positions, rope_forward):
            self.capture = (key.clone(), positions.clone(), rope_forward)

    handler = FakeHandler()
    monkeypatch.setattr(vllm_model, "get_attention_handler", lambda: handler)

    def original_rope(self, positions, query, key):
        del self, positions
        key.add_(1)
        return query, key

    monkeypatch.setattr(vllm_model, "_ORIGINAL_ROTARY_EMBEDDING_CALL", original_rope)
    rope = type("Rope", (), {"_forward_method": object()})()
    key = torch.zeros(3, 8)
    positions = torch.arange(3)
    query = torch.zeros_like(key)
    _, rotated = vllm_model._shadowkv_rotary_embedding_call(
        rope, positions, query, key
    )

    assert handler.capture is not None
    captured_key, captured_positions, captured_forward = handler.capture
    assert torch.count_nonzero(captured_key) == 0
    torch.testing.assert_close(captured_positions, positions)
    assert captured_forward is rope._forward_method
    torch.testing.assert_close(rotated, torch.ones_like(key))


def test_vllm_rope_hook_captures_current_rope_cache_after_dispatch(monkeypatch):
    from sparse_frontier.modelling.models import vllm_model

    monkeypatch.setenv("SF_ATTENTION_NAME", "shadowkv")

    class FakeHandler:
        def __init__(self):
            self.metadata = None

        def capture_pre_rope(self, key, positions, rope_forward):
            del key, positions, rope_forward

        def capture_rope_metadata(self, rotary_embedding):
            self.metadata = rotary_embedding.cos_sin_cache

    handler = FakeHandler()
    monkeypatch.setattr(vllm_model, "get_attention_handler", lambda: handler)

    def original_rope(self, positions, query, key):
        del positions
        self.cos_sin_cache = torch.ones(8, 4, dtype=torch.bfloat16)
        key.add_(1)
        return query, key

    monkeypatch.setattr(vllm_model, "_ORIGINAL_ROTARY_EMBEDDING_CALL", original_rope)
    rope = type(
        "Rope",
        (),
        {
            "_forward_method": object(),
            "cos_sin_cache": torch.zeros(8, 4, dtype=torch.bfloat16),
            "head_size": 4,
            "rotary_dim": 4,
            "is_neox_style": True,
        },
    )()
    key = torch.zeros(3, 8)
    positions = torch.arange(3)
    query = torch.zeros_like(key)
    vllm_model._shadowkv_rotary_embedding_call(rope, positions, query, key)

    assert handler.metadata is rope.cos_sin_cache
    torch.testing.assert_close(handler.metadata, torch.ones(8, 4, dtype=torch.bfloat16))
