"""Dense side-effect capture for position-transported empirical queries."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import Tensor


RoPEForward = Callable[..., tuple[Tensor, Tensor | None]]


@dataclass(frozen=True)
class CaptureContext:
    sequence_id_hash: int
    task: str
    task_index: int
    split: str
    prompt_length: int
    stratum_id: int

    def __post_init__(self) -> None:
        if self.prompt_length < 1:
            raise ValueError("capture prompt length must be positive")
        if self.task_index < 0:
            raise ValueError("capture task index must be nonnegative")
        if not self.task or not self.split:
            raise ValueError("capture task and split cannot be empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CaptureContext":
        if not isinstance(payload, dict):
            raise ValueError("capture context must be a JSON object")
        expected = {field.name for field in fields(cls)}
        actual = set(payload)
        unexpected = sorted(actual - expected)
        missing = sorted(expected - actual)
        if unexpected:
            raise ValueError(f"capture context has unexpected fields: {unexpected}")
        if missing:
            raise ValueError(f"capture context is missing fields: {missing}")
        return cls(**payload)


class PromptQueryReservoir:
    """Deterministic per-layer/Q-head reservoir for prompt pre-RoPE queries.

    The reservoir is deliberately independent of keys, generated states, and
    task labels.  It therefore represents only a compact empirical measure of
    the current prompt query stream.  Stored rows retain the pre-RoPE query
    representation and their absolute prompt positions for later transport.
    """

    def __init__(
        self,
        num_layers: int,
        num_q_heads: int,
        head_dim: int,
        size: int,
        seed: int,
    ) -> None:
        positive = {
            "num_layers": num_layers,
            "num_q_heads": num_q_heads,
            "head_dim": head_dim,
            "size": size,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"prompt reservoir {name} must be a positive integer")
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.size = size
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.queries = torch.zeros(
            num_layers, num_q_heads, size, head_dim, dtype=torch.bfloat16
        )
        self.priority = torch.full(
            (num_layers, num_q_heads, size), float("-inf"), dtype=torch.float32
        )
        self.positions = torch.full(
            (num_layers, num_q_heads, size), -1, dtype=torch.int32
        )

    @torch.no_grad()
    def update(self, layer: int, queries: Tensor, positions: Tensor) -> None:
        """Merge one ``[q_heads,tokens,head_dim]`` prompt batch."""

        if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < self.num_layers:
            raise ValueError(f"prompt reservoir layer {layer!r} is out of range")
        if (
            queries.ndim != 3
            or queries.shape[0] != self.num_q_heads
            or queries.shape[2] != self.head_dim
            or queries.shape[1] < 1
        ):
            raise ValueError(
                "prompt reservoir query batch must have shape "
                "[q_heads,tokens,head_dim]"
            )
        if positions.ndim != 1 or positions.numel() != queries.shape[1]:
            raise ValueError("prompt reservoir positions do not match query tokens")
        batch = queries.detach().to(device="cpu", dtype=torch.bfloat16)
        batch_positions = positions.detach().to(device="cpu", dtype=torch.int32)
        batch_priority = torch.rand(
            self.num_q_heads,
            queries.shape[1],
            generator=self.generator,
            dtype=torch.float32,
        )
        for head in range(self.num_q_heads):
            candidate_priority = torch.cat(
                [self.priority[layer, head], batch_priority[head]]
            )
            candidate_queries = torch.cat(
                [self.queries[layer, head], batch[head]], dim=0
            )
            candidate_positions = torch.cat(
                [self.positions[layer, head], batch_positions], dim=0
            )
            selected_priority, selected = torch.topk(
                candidate_priority, k=self.size, largest=True, sorted=True
            )
            self.priority[layer, head] = selected_priority
            self.queries[layer, head] = candidate_queries[selected]
            self.positions[layer, head] = candidate_positions[selected]

    def complete_layer(self, layer: int) -> None:
        if torch.any(self.positions[layer] < 0):
            raise RuntimeError(
                "prompt query reservoir did not receive enough tokens for "
                f"layer {layer}"
            )

    def complete(self) -> None:
        for layer in range(self.num_layers):
            self.complete_layer(layer)


class QueryCaptureCollector:
    """Collect one request at a time in a vLLM worker process.

    Query capture is calibration-only and synchronous by design.  Prompt keys
    are written once per selected layer, while decode queries and generated
    keys are written once per step after the final model layer.
    """

    def __init__(
        self,
        capture_dir: str | Path,
        context_path: str | Path,
        num_layers: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        tp_rank: int,
        capture_layers: Sequence[int] | None = None,
        capture_values: bool = False,
        capture_prompt_keys: bool = True,
        capture_prompt_queries: bool = False,
        prompt_query_samples: int = 16,
        prompt_query_seed: int = 1043,
        composition_tolerance: float = 0.05,
    ) -> None:
        positive = {
            "num_layers": num_layers,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"capture {name} must be a positive integer")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("capture query heads must be divisible by KV heads")
        if not isinstance(tp_rank, int) or isinstance(tp_rank, bool) or tp_rank < 0:
            raise ValueError("capture TP rank must be a nonnegative integer")
        if not composition_tolerance >= 0:
            raise ValueError("capture composition tolerance must be nonnegative")
        selected = tuple(range(num_layers)) if capture_layers is None else tuple(capture_layers)
        if len(set(selected)) != len(selected) or any(
            not isinstance(layer, int) or layer < 0 or layer >= num_layers
            for layer in selected
        ):
            raise ValueError("capture layers must be unique valid model-layer indices")

        self.capture_dir = Path(capture_dir)
        self.context_path = Path(context_path)
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tp_rank = tp_rank
        self.capture_layers = selected
        self._capture_layer_set = set(selected)
        self.capture_values = bool(capture_values)
        self.capture_prompt_keys = bool(capture_prompt_keys)
        if not isinstance(prompt_query_samples, int) or isinstance(prompt_query_samples, bool) or prompt_query_samples < 1:
            raise ValueError("prompt query samples must be a positive integer")
        self.capture_prompt_queries = bool(capture_prompt_queries)
        self.prompt_query_samples = int(prompt_query_samples)
        self.prompt_query_seed = int(prompt_query_seed)
        self.composition_tolerance = float(composition_tolerance)

        self.context: CaptureContext | None = None
        self._request_dir: Path | None = None
        self._pending_pre_query: Tensor | None = None
        self._pending_positions: Tensor | None = None
        self._pending_rope: RoPEForward | None = None
        self._step_layers: dict[int, dict[str, Tensor | float]] = {}
        self._decode_offset = 0
        self._implicit_layer = 0
        self._prompt_query_reservoir: PromptQueryReservoir | None = None

    def _read_context(self) -> CaptureContext:
        if not self.context_path.is_file():
            raise RuntimeError(
                f"query capture context is missing: {self.context_path}"
            )
        try:
            payload = json.loads(self.context_path.read_text(encoding="utf-8"))
            return CaptureContext.from_dict(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"query capture context is invalid: {exc}") from exc

    def reset_request(self, context: CaptureContext) -> None:
        self.context = context
        self._request_dir = self.capture_dir / f"request_{context.sequence_id_hash:016x}"
        self._request_dir.mkdir(parents=True, exist_ok=True)
        self._pending_pre_query = None
        self._pending_positions = None
        self._pending_rope = None
        self._step_layers.clear()
        self._decode_offset = 0
        self._implicit_layer = 0
        self._prompt_query_reservoir = (
            PromptQueryReservoir(
                self.num_layers,
                self.num_q_heads,
                self.head_dim,
                self.prompt_query_samples,
                self.prompt_query_seed,
            )
            if self.capture_prompt_queries
            else None
        )

    def _ensure_request_context(self) -> None:
        context = self._read_context()
        if self.context is None or self.context.sequence_id_hash != context.sequence_id_hash:
            self.reset_request(context)

    def _write_prompt_query_shard(self, layer_idx: int) -> None:
        if not self.capture_prompt_queries:
            return
        if self.context is None or self._request_dir is None:
            raise RuntimeError("prompt query capture has no active request context")
        reservoir = self._prompt_query_reservoir
        if reservoir is None:
            raise RuntimeError("prompt query reservoir was not initialized")
        reservoir.complete_layer(layer_idx)
        self._atomic_torch_save(
            {
                "prompt_pre_rope_queries": reservoir.queries[layer_idx].clone(),
                "prompt_positions": reservoir.positions[layer_idx].clone(),
                "layer_idx": layer_idx,
                "representation": "q_norm_pre_rope_pre_scale",
                "sampling": {
                    "method": "uniform_priority_reservoir",
                    "samples_per_q_head": self.prompt_query_samples,
                    "seed": self.prompt_query_seed,
                },
                "metadata": asdict(self.context),
            },
            self._request_dir
            / f"prompt_queries_layer_{layer_idx:03d}_rank_{self.tp_rank:03d}.pt",
        )

    def _query_view(self, query: Tensor) -> Tensor:
        if query.ndim == 2:
            if query.shape[-1] != self.num_q_heads * self.head_dim:
                raise RuntimeError(
                    "query capture cannot reshape flattened query: "
                    f"shape={tuple(query.shape)}, heads={self.num_q_heads}, dim={self.head_dim}"
                )
            return query.view(query.shape[0], self.num_q_heads, self.head_dim)
        if query.ndim == 3 and query.shape[1:] == (self.num_q_heads, self.head_dim):
            return query
        raise RuntimeError(
            "query capture expects [tokens, q_heads*dim] or [tokens, q_heads, dim], "
            f"got {tuple(query.shape)}"
        )

    def _key_view(self, key: Tensor) -> Tensor:
        if key.ndim == 2:
            if key.shape[-1] != self.num_kv_heads * self.head_dim:
                raise RuntimeError("query capture cannot reshape flattened key")
            return key.view(key.shape[0], self.num_kv_heads, self.head_dim)
        if key.ndim == 3 and key.shape[1:] == (self.num_kv_heads, self.head_dim):
            return key
        raise RuntimeError(
            "query capture expects [tokens, kv_heads*dim] or [tokens, kv_heads, dim], "
            f"got {tuple(key.shape)}"
        )

    def capture_pre_rope(
        self,
        query: Tensor,
        positions: Tensor,
        rope_forward: RoPEForward,
    ) -> None:
        """Clone a one-token query before vLLM's in-place rotary operation."""

        if positions.ndim != 1:
            raise RuntimeError(
                f"query capture requires scalar positions, got {tuple(positions.shape)}"
            )
        query_view = self._query_view(query)
        if query_view.shape[0] != positions.numel():
            raise RuntimeError("query capture position and token counts differ")
        if query_view.shape[0] != 1:
            if self.capture_prompt_queries:
                # vLLM performs a context-free dummy/profile forward before
                # the first user request.  There is no valid request source
                # to capture in that pass; the real prefill path still fails
                # closed in capture_attention if its context is absent.
                if not self.context_path.is_file():
                    self._pending_pre_query = None
                    self._pending_positions = None
                    self._pending_rope = None
                    return
                self._ensure_request_context()
                if self.context is None:
                    raise RuntimeError("prompt query capture has no request context")
                if query_view.shape[0] != self.context.prompt_length:
                    raise RuntimeError(
                        "prompt query capture did not receive the full prompt; "
                        f"context={self.context.prompt_length}, tensor={query_view.shape[0]}"
                    )
                reservoir = self._prompt_query_reservoir
                if reservoir is None:
                    raise RuntimeError("prompt query reservoir was not initialized")
                layer_idx = self._resolve_layer(None)
                if layer_idx in self._capture_layer_set:
                    reservoir.update(layer_idx, query_view.transpose(0, 1), positions)
            self._pending_pre_query = None
            self._pending_positions = None
            self._pending_rope = None
            return
        self._pending_pre_query = query_view[0].detach().clone().contiguous()
        self._pending_positions = positions.detach().clone().contiguous()
        self._pending_rope = rope_forward

    def _apply_query_rope(self, query: Tensor, position: int, rope: RoPEForward) -> Tensor:
        positions = torch.full(
            (query.shape[0],), position, device=query.device, dtype=torch.long
        )
        rotated, _ = rope(positions, query.clone().contiguous(), None)
        if rotated is None or rotated.shape != query.shape:
            raise RuntimeError("query capture RoPE dispatch returned an invalid query")
        return rotated

    def _composition_limit(self, query: Tensor) -> float:
        """Allow two BF16 RoPE applications the rounding error they imply."""
        dtype_tolerance = 0.0
        if query.dtype.is_floating_point:
            scale = max(1.0, float(query.detach().float().abs().max().item()))
            dtype_tolerance = 2.0 * torch.finfo(query.dtype).eps * scale
        return max(self.composition_tolerance, dtype_tolerance)

    @staticmethod
    def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _resolve_layer(self, layer_idx: int | None) -> int:
        if layer_idx is not None:
            if not 0 <= layer_idx < self.num_layers:
                raise RuntimeError(f"query capture layer {layer_idx} is outside the model")
            return layer_idx
        return self._implicit_layer

    def _advance_implicit_layer(self, explicit: bool) -> None:
        if not explicit:
            self._implicit_layer = (self._implicit_layer + 1) % self.num_layers

    def capture_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor | None,
        is_prefill: bool,
        layer_idx: int | None = None,
    ) -> None:
        """Capture post-RoPE attention inputs without changing their storage."""

        explicit_layer = layer_idx is not None
        resolved_layer = self._resolve_layer(layer_idx)
        query_view = self._query_view(query)
        key_view = self._key_view(key)

        if is_prefill:
            if resolved_layer == 0:
                context = self._read_context()
                if self.context is None or self.context.sequence_id_hash != context.sequence_id_hash:
                    self.reset_request(context)
            if self.context is None or self._request_dir is None:
                raise RuntimeError("query capture prefill started without request context")
            if query_view.shape[0] != self.context.prompt_length:
                raise RuntimeError(
                    "query capture prompt length mismatch: "
                    f"context={self.context.prompt_length}, tensor={query_view.shape[0]}"
                )
            if resolved_layer in self._capture_layer_set and self.capture_prompt_keys:
                payload: dict[str, Any] = {
                    "keys": key_view.detach().to(device="cpu", dtype=torch.bfloat16),
                    "layer_idx": resolved_layer,
                    "metadata": asdict(self.context),
                }
                if self.capture_values:
                    if value is None:
                        raise RuntimeError("query capture requested values but received none")
                    payload["values"] = self._key_view(value).detach().to(
                        device="cpu", dtype=torch.bfloat16
                    )
                self._atomic_torch_save(
                    payload,
                    self._request_dir
                    / f"prompt_layer_{resolved_layer:03d}_rank_{self.tp_rank:03d}.pt",
                )
            if resolved_layer in self._capture_layer_set and self.capture_prompt_queries:
                self._write_prompt_query_shard(resolved_layer)
            self._advance_implicit_layer(explicit_layer)
            return

        if self.context is None or self._request_dir is None:
            raise RuntimeError("query capture decode started before dense prefill context")
        if query_view.shape[0] != 1 or key_view.shape[0] != 1:
            raise RuntimeError("query capture decode supports exactly one token")
        if (
            self._pending_pre_query is None
            or self._pending_positions is None
            or self._pending_rope is None
        ):
            raise RuntimeError("query capture has no paired pre-RoPE decode query")

        absolute_position = int(self._pending_positions.item())
        decode_offset = absolute_position - self.context.prompt_length
        origin = self._apply_query_rope(
            self._pending_pre_query, decode_offset, self._pending_rope
        )
        reconstructed = self._apply_query_rope(
            origin, self.context.prompt_length, self._pending_rope
        )
        post = query_view[0]
        composition_error = float(
            (reconstructed.float() - post.float()).abs().max().item()
        )
        composition_limit = self._composition_limit(post)
        if composition_error > composition_limit:
            raise RuntimeError(
                "query capture RoPE composition check failed: "
                f"max_abs={composition_error}, tolerance={composition_limit}"
            )
        if decode_offset != self._decode_offset:
            raise RuntimeError(
                "query capture decode offset mismatch: "
                f"expected={self._decode_offset}, observed={decode_offset}"
            )
        if resolved_layer in self._capture_layer_set:
            layer_payload: dict[str, Tensor | float] = {
                "prompt_origin_query": origin.detach().to(
                    device="cpu", dtype=torch.bfloat16
                ),
                "post_rope_query": post.detach().to(
                    device="cpu", dtype=torch.bfloat16
                ),
                "generated_key": key_view.detach().to(
                    device="cpu", dtype=torch.bfloat16
                ),
                "composition_max_abs": composition_error,
                "composition_tolerance": composition_limit,
            }
            if self.capture_values:
                if value is None:
                    raise RuntimeError("query capture requested values but received none")
                layer_payload["generated_value"] = self._key_view(value).detach().to(
                    device="cpu", dtype=torch.bfloat16
                )
            self._step_layers[resolved_layer] = layer_payload

        self._pending_pre_query = None
        self._pending_positions = None
        self._pending_rope = None

        if resolved_layer == self.num_layers - 1:
            missing = [layer for layer in self.capture_layers if layer not in self._step_layers]
            if missing:
                raise RuntimeError(f"query capture decode step is missing layers {missing}")
            ordered = [self._step_layers[layer] for layer in self.capture_layers]
            metadata = asdict(self.context)
            metadata.update(
                {
                    "decode_offset": self._decode_offset,
                    "absolute_position": absolute_position,
                    "composition_max_abs": max(
                        float(item["composition_max_abs"]) for item in ordered
                    ),
                    "composition_tolerance": max(
                        float(item["composition_tolerance"]) for item in ordered
                    ),
                    "tp_rank": self.tp_rank,
                }
            )
            payload = {
                "layer_ids": torch.tensor(self.capture_layers, dtype=torch.int64),
                "prompt_origin_queries": torch.stack(
                    [item["prompt_origin_query"] for item in ordered], dim=0
                ),
                "post_rope_queries": torch.stack(
                    [item["post_rope_query"] for item in ordered], dim=0
                ),
                "generated_keys": torch.stack(
                    [item["generated_key"] for item in ordered], dim=0
                ),
                "metadata": metadata,
            }
            if self.capture_values:
                payload["generated_values"] = torch.stack(
                    [item["generated_value"] for item in ordered], dim=0
                )
            self._atomic_torch_save(
                payload,
                self._request_dir
                / f"step_{self._decode_offset:03d}_rank_{self.tp_rank:03d}.pt",
            )
            self._step_layers.clear()
            self._decode_offset += 1
        self._advance_implicit_layer(explicit_layer)
