import json
from pathlib import Path

import pytest
import torch

from sparse_frontier.modelling.attention.query_capture import (
    CaptureContext,
    PromptQueryReservoir,
    QueryCaptureCollector,
)


def _context(path: Path, prompt_length=8):
    payload = {
        "sequence_id_hash": 1234,
        "task": "niah_single",
        "task_index": 0,
        "split": "calibration",
        "prompt_length": prompt_length,
        "stratum_id": 2,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return CaptureContext.from_dict(payload)


def _additive_rope(positions, query, key=None):
    delta = positions.to(query.dtype).view(-1, 1)
    query.add_(delta)
    if key is not None:
        key.add_(delta)
    return query, key


def _run_decode_layer(collector, layer_idx, offset, prompt_length=8):
    pre = torch.arange(8, dtype=torch.float32).view(1, 8) + layer_idx
    position = torch.tensor([prompt_length + offset])
    collector.capture_pre_rope(pre, position, _additive_rope)
    post = pre.clone()
    _additive_rope(position, post)
    key = torch.full((1, 1, 4), layer_idx + offset, dtype=torch.float32)
    collector.capture_attention(post, key, None, is_prefill=False, layer_idx=layer_idx)


def test_collector_writes_prompt_keys_and_prompt_origin_decode_queries(tmp_path):
    context_path = tmp_path / "context.json"
    _context(context_path)
    collector = QueryCaptureCollector(
        capture_dir=tmp_path / "capture",
        context_path=context_path,
        num_layers=2,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        tp_rank=0,
        capture_layers=(0, 1),
        composition_tolerance=1e-6,
    )

    query = torch.zeros(8, 2, 4)
    key = torch.arange(32, dtype=torch.float32).view(8, 1, 4)
    collector.capture_attention(query, key, None, is_prefill=True, layer_idx=0)
    collector.capture_attention(query, key + 1, None, is_prefill=True, layer_idx=1)

    _run_decode_layer(collector, layer_idx=0, offset=0)
    _run_decode_layer(collector, layer_idx=1, offset=0)
    _run_decode_layer(collector, layer_idx=0, offset=1)
    _run_decode_layer(collector, layer_idx=1, offset=1)

    request_dir = tmp_path / "capture" / "request_00000000000004d2"
    prompt = torch.load(
        request_dir / "prompt_layer_000_rank_000.pt", weights_only=True
    )
    assert prompt["keys"].dtype == torch.bfloat16
    assert prompt["keys"].shape == (8, 1, 4)

    step0 = torch.load(request_dir / "step_000_rank_000.pt", weights_only=True)
    step1 = torch.load(request_dir / "step_001_rank_000.pt", weights_only=True)
    torch.testing.assert_close(step0["layer_ids"], torch.tensor([0, 1]))
    assert step0["prompt_origin_queries"].shape == (2, 2, 4)
    assert step0["metadata"]["decode_offset"] == 0
    assert step1["metadata"]["decode_offset"] == 1
    assert step0["metadata"]["composition_max_abs"] == pytest.approx(0.0)
    expected_origin = torch.arange(8, dtype=torch.float32).view(2, 4) + 1
    torch.testing.assert_close(step1["prompt_origin_queries"][0].float(), expected_origin)


def test_collector_rejects_missing_context(tmp_path):
    collector = QueryCaptureCollector(
        capture_dir=tmp_path / "capture",
        context_path=tmp_path / "missing.json",
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        tp_rank=0,
    )
    with pytest.raises(RuntimeError, match="context"):
        collector.capture_attention(
            torch.zeros(4, 2, 4),
            torch.zeros(4, 1, 4),
            None,
            is_prefill=True,
            layer_idx=0,
        )


def test_collector_rejects_noncompositional_rope(tmp_path):
    context_path = tmp_path / "context.json"
    _context(context_path, prompt_length=4)
    collector = QueryCaptureCollector(
        capture_dir=tmp_path / "capture",
        context_path=context_path,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        tp_rank=0,
        composition_tolerance=1e-6,
    )
    collector.capture_attention(
        torch.zeros(4, 2, 4), torch.zeros(4, 1, 4), None, True, 0
    )

    def squared_rope(positions, query, key=None):
        delta = positions.to(query.dtype).square().view(-1, 1)
        query.add_(delta)
        return query, key

    pre = torch.zeros(1, 8)
    position = torch.tensor([5])
    collector.capture_pre_rope(pre, position, squared_rope)
    post = pre.clone()
    squared_rope(position, post)
    with pytest.raises(RuntimeError, match="compos"):
        collector.capture_attention(
            post, torch.zeros(1, 1, 4), None, False, 0
        )


def test_capture_context_rejects_unknown_and_missing_fields():
    payload = {
        "sequence_id_hash": 1,
        "task": "x",
        "task_index": 0,
        "split": "calibration",
        "prompt_length": 8,
        "stratum_id": 0,
    }
    assert CaptureContext.from_dict(payload).prompt_length == 8
    with pytest.raises(ValueError, match="unexpected"):
        CaptureContext.from_dict({**payload, "extra": 1})
    missing = dict(payload)
    missing.pop("task")
    with pytest.raises(ValueError, match="missing"):
        CaptureContext.from_dict(missing)


def test_composition_limit_accounts_for_bfloat16_rounding(tmp_path):
    collector = QueryCaptureCollector(
        capture_dir=tmp_path / "capture",
        context_path=tmp_path / "context.json",
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        tp_rank=0,
        composition_tolerance=0.05,
    )
    query = torch.full((2, 4), 18.0, dtype=torch.bfloat16)
    assert collector._composition_limit(query) >= 0.25


def test_dense_capture_forward_preserves_original_output(monkeypatch, tmp_path):
    from sparse_frontier.modelling.models import vllm_model

    monkeypatch.setenv("SF_QUERY_CAPTURE_DIR", str(tmp_path))
    monkeypatch.delenv("SF_ATTENTION_NAME", raising=False)

    class FakeCollector:
        def __init__(self):
            self.calls = []

        def capture_attention(self, query, key, value, is_prefill, layer_idx=None):
            self.calls.append((query.clone(), key.clone(), value.clone(), is_prefill, layer_idx))

    collector = FakeCollector()
    monkeypatch.setattr(vllm_model, "_get_query_capture_collector", lambda: collector)

    def original(self, layer, query, key, value, kv_cache, metadata, output, *rest):
        del self, layer, key, value, kv_cache, metadata, rest
        output.copy_(query * 3)
        return output

    monkeypatch.setattr(vllm_model, "_ORIGINAL_FLASH_ATTENTION_FORWARD", original)
    query = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
    key = torch.zeros(1, 1, 4)
    value = torch.ones_like(key)
    output = torch.empty_like(query)
    fake_self = type("Attention", (), {"sliding_window": (-1, -1)})()
    result = vllm_model.vllm_patched_forward(
        fake_self,
        object(),
        query,
        key,
        value,
        torch.empty(0),
        object(),
        output,
    )

    torch.testing.assert_close(result, query * 3)
    assert len(collector.calls) == 1
    assert collector.calls[0][3] is False


def test_rotary_hook_captures_query_before_in_place_mutation(monkeypatch):
    from sparse_frontier.modelling.models import vllm_model

    monkeypatch.setenv("SF_QUERY_CAPTURE_DIR", "/tmp/capture")
    monkeypatch.delenv("SF_ATTENTION_NAME", raising=False)

    class FakeCollector:
        def __init__(self):
            self.query = None

        def capture_pre_rope(self, query, positions, rope_forward):
            del positions, rope_forward
            self.query = query.clone()

    collector = FakeCollector()
    monkeypatch.setattr(vllm_model, "_get_query_capture_collector", lambda: collector)

    def original(self, positions, query, key):
        del self, positions
        query.add_(7)
        return query, key

    monkeypatch.setattr(vllm_model, "_ORIGINAL_ROTARY_EMBEDDING_CALL", original)
    rope = type("Rope", (), {"_forward_method": _additive_rope})()
    query = torch.zeros(1, 8)
    _, _ = vllm_model._shadowkv_rotary_embedding_call(
        rope, torch.tensor([8]), query, torch.zeros(1, 4)
    )
    assert collector.query is not None
    assert torch.count_nonzero(collector.query) == 0
    torch.testing.assert_close(query, torch.full_like(query, 7))


def test_queries_only_capture_skips_large_prompt_key_shards(tmp_path):
    context_path = tmp_path / "context.json"
    _context(context_path, prompt_length=4)
    collector = QueryCaptureCollector(
        capture_dir=tmp_path / "capture",
        context_path=context_path,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        tp_rank=0,
        capture_prompt_keys=False,
    )
    collector.capture_attention(
        torch.zeros(4, 2, 4), torch.zeros(4, 1, 4), None, True, 0
    )
    assert not list((tmp_path / "capture").rglob("prompt*.pt"))


def test_prompt_query_reservoir_is_deterministic_and_validates_completion():
    first = PromptQueryReservoir(
        num_layers=1, num_q_heads=2, head_dim=3, size=2, seed=17
    )
    second = PromptQueryReservoir(
        num_layers=1, num_q_heads=2, head_dim=3, size=2, seed=17
    )
    queries = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    positions = torch.arange(5, dtype=torch.int32)
    first.update(0, queries, positions)
    second.update(0, queries, positions)
    first.complete()
    second.complete()
    torch.testing.assert_close(first.queries, second.queries)
    torch.testing.assert_close(first.positions, second.positions)
    assert first.queries.dtype == torch.bfloat16
    assert first.positions.shape == (1, 2, 2)
    assert torch.all((first.positions >= 0) & (first.positions < 5))


def test_collector_writes_prompt_pre_rope_query_reservoir(tmp_path):
    context_path = tmp_path / "context.json"
    _context(context_path, prompt_length=4)
    collector = QueryCaptureCollector(
        capture_dir=tmp_path / "capture",
        context_path=context_path,
        num_layers=2,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        tp_rank=0,
        capture_layers=(0, 1),
        capture_prompt_keys=False,
        capture_prompt_queries=True,
        prompt_query_samples=2,
        prompt_query_seed=19,
    )
    query = torch.arange(4 * 2 * 4, dtype=torch.float32).view(4, 2, 4)
    positions = torch.arange(4, dtype=torch.int32)
    collector.capture_pre_rope(query, positions, _additive_rope)
    collector.capture_attention(
        query.clone(), torch.zeros(4, 1, 4), None, is_prefill=True
    )
    collector.capture_pre_rope(query + 100, positions, _additive_rope)
    collector.capture_attention(
        query.clone(), torch.zeros(4, 1, 4), None, is_prefill=True
    )
    request_dir = tmp_path / "capture" / "request_00000000000004d2"
    payload = torch.load(
        request_dir / "prompt_queries_layer_000_rank_000.pt", weights_only=True
    )
    assert payload["prompt_pre_rope_queries"].shape == (2, 2, 4)
    assert payload["prompt_positions"].shape == (2, 2)
    assert payload["metadata"]["prompt_length"] == 4
    assert payload["representation"] == "q_norm_pre_rope_pre_scale"


def test_prompt_query_capture_requires_enough_prompt_tokens(tmp_path):
    context_path = tmp_path / "context.json"
    _context(context_path, prompt_length=2)
    collector = QueryCaptureCollector(
        capture_dir=tmp_path / "capture",
        context_path=context_path,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        tp_rank=0,
        capture_layers=(0,),
        capture_prompt_queries=True,
        prompt_query_samples=3,
    )
    collector.capture_pre_rope(
        torch.zeros(2, 2, 4), torch.arange(2), _additive_rope
    )
    with pytest.raises(RuntimeError, match="enough"):
        collector.capture_attention(
            torch.zeros(2, 2, 4), torch.zeros(2, 1, 4), None, is_prefill=True
        )
