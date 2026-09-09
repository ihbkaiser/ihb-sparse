from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sparsevllm.config import Config
from sparsevllm.distributed import ParallelContext
from sparsevllm.engine.cache_manager.base import (
    AttentionCacheWrite,
    ExplicitKVWrite,
)
from sparsevllm.engine.cache_manager.quest import QuestCacheManager
from sparsevllm.engine.cache_manager.storage import ExplicitKVStorage
from sparsevllm.method_registry import normalize_sparse_method
from sparsevllm.operators.quest_selection import (
    QuestPageSelectionOpSpec,
    resolve_quest_page_selection_provider,
)
from sparsevllm.utils.context import get_context
from sparsevllm.utils.profiler import profiler


@dataclass(frozen=True)
class QueryRobustAsset:
    """Validated global QR vertices sliced to one tensor-parallel rank."""

    vertices: torch.Tensor
    num_valid_vertices: torch.Tensor
    meta: dict[str, Any]


@dataclass(frozen=True)
class QueryRobustSummary:
    """Stored page summaries and solver diagnostics."""

    landmark: torch.Tensor
    bias: torch.Tensor
    epsilon: torch.Tensor
    dual: torch.Tensor
    gap: torch.Tensor
    errors: torch.Tensor


def _as_tensor(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
        if name == "vertices" and tensor.dtype == torch.uint16:
            # NumPy has no native BF16 dtype; NPZ assets store raw BF16 bits.
            return tensor.view(torch.bfloat16)
        return tensor
    raise TypeError(f"Query-Robust asset field {name!r} must be a tensor or ndarray.")


def _load_asset_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            payload: dict[str, Any] = {
                "vertices": data["vertices"],
                "num_valid_vertices": data["num_valid_vertices"],
            }
            if "meta" in data:
                raw_meta = data["meta"].item() if data["meta"].ndim == 0 else data["meta"]
                if isinstance(raw_meta, bytes):
                    raw_meta = raw_meta.decode("utf-8")
                payload["meta"] = (
                    json.loads(str(raw_meta)) if isinstance(raw_meta, str) else raw_meta
                )
            return payload
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_query_robust_asset(
    path: str | Path,
    *,
    num_layers: int,
    global_num_kv_heads: int,
    head_dim: int,
    num_vertices: int,
    tensor_parallel_rank: int,
    tensor_parallel_size: int,
    expected_model_id: str | None = None,
    expected_model_fingerprint: str | None = None,
    expected_rope_config: Any | None = None,
    device: torch.device | None = None,
) -> QueryRobustAsset:
    """Load, validate, and TP-slice a QR vertex asset."""

    payload = _load_asset_payload(path)
    if not isinstance(payload, dict):
        raise TypeError("Query-Robust asset must contain a mapping payload.")
    vertices = _as_tensor(payload.get("vertices"), name="vertices")
    valid = _as_tensor(payload.get("num_valid_vertices"), name="num_valid_vertices")
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        raise TypeError("Query-Robust asset meta must be a mapping.")
    if vertices.ndim != 4:
        raise ValueError(
            "Query-Robust vertices must have shape [layers, global_kv_heads, M, D], "
            f"got {tuple(vertices.shape)}."
        )
    if vertices.dtype != torch.bfloat16:
        raise TypeError(
            "Query-Robust vertex assets must store vertices as BF16; "
            f"got {vertices.dtype}."
        )
    if valid.ndim != 2:
        raise ValueError(
            "Query-Robust num_valid_vertices must have shape [layers, global_kv_heads], "
            f"got {tuple(valid.shape)}."
        )
    actual_layers, actual_global_heads, actual_vertices, actual_dim = map(int, vertices.shape)
    if tuple(valid.shape) != (actual_layers, actual_global_heads):
        raise ValueError(
            "Query-Robust validity shape does not match vertices: "
            f"vertices={tuple(vertices.shape)} valid={tuple(valid.shape)}."
        )
    expected = {
        "num_layers": int(num_layers),
        "global_num_kv_heads": int(global_num_kv_heads),
        "head_dim": int(head_dim),
        "num_vertices": int(num_vertices),
    }
    actual = {
        "num_layers": actual_layers,
        "global_num_kv_heads": actual_global_heads,
        "head_dim": actual_dim,
        "num_vertices": actual_vertices,
    }
    for name, wanted in expected.items():
        if actual[name] != wanted:
            raise ValueError(
                f"Query-Robust asset {name} mismatch: expected={wanted} got={actual[name]}."
            )
    if "tp_world_size" not in meta:
        raise ValueError(
            "Query-Robust asset metadata must declare tp_world_size so the KV-head "
            "partition can be checked explicitly."
        )
    meta_tp = int(meta["tp_world_size"])
    if meta_tp != int(tensor_parallel_size):
        raise ValueError(
            "Query-Robust asset TP partition does not match the runtime: "
            f"asset_tp_world_size={meta_tp} runtime_tp_world_size={tensor_parallel_size}."
        )
    rank = int(tensor_parallel_rank)
    tp_size = int(tensor_parallel_size)
    if not 0 <= rank < tp_size:
        raise ValueError(f"Invalid Query-Robust TP rank={rank} for size={tp_size}.")
    if actual_global_heads % tp_size:
        raise ValueError(
            "Query-Robust global KV heads must be divisible by asset TP size: "
            f"heads={actual_global_heads} tp={tp_size}."
        )

    model_id = meta.get("model_id")
    if expected_model_id is not None and model_id is None:
        raise ValueError(
            "Query-Robust asset metadata must declare model_id for runtime validation."
        )
    if expected_model_id is not None and model_id is not None:
        expected_id = str(expected_model_id).rstrip("/").split("/")[-1]
        actual_id = str(model_id).rstrip("/").split("/")[-1]
        if expected_id != actual_id:
            raise ValueError(
                "Query-Robust asset model_id mismatch: "
                f"expected={expected_model_id!r} asset={model_id!r}."
            )
    asset_fingerprint = meta.get("model_fingerprint")
    if asset_fingerprint is not None:
        if expected_model_fingerprint is None:
            raise ValueError(
                "Query-Robust asset declares model_fingerprint but the runtime "
                "did not provide query_robust_model_fingerprint."
            )
        if str(asset_fingerprint) != str(expected_model_fingerprint):
            raise ValueError(
                "Query-Robust asset model_fingerprint mismatch: "
                f"expected={expected_model_fingerprint!r} asset={asset_fingerprint!r}."
            )
    elif expected_model_fingerprint is not None:
        raise ValueError(
            "Query-Robust asset metadata must declare model_fingerprint when the "
            "runtime config supplies query_robust_model_fingerprint."
        )
    asset_rope = meta.get("rope_config")
    if expected_rope_config is not None and asset_rope is None:
        raise ValueError(
            "Query-Robust asset metadata must declare rope_config for runtime validation."
        )
    if asset_rope is not None and expected_rope_config is not None:
        if json.dumps(asset_rope, sort_keys=True, default=str) != json.dumps(
            expected_rope_config, sort_keys=True, default=str
        ):
            raise ValueError("Query-Robust asset RoPE configuration does not match runtime.")
    for meta_name, actual_value in (
        ("num_layers", actual_layers),
        ("num_kv_heads", actual_global_heads),
        ("head_dim", actual_dim),
        ("M", actual_vertices),
    ):
        if meta_name in meta and int(meta[meta_name]) != actual_value:
            raise ValueError(
                f"Query-Robust asset metadata {meta_name} disagrees with tensor shape: "
                f"meta={meta[meta_name]} shape={actual_value}."
            )

    valid = valid.to(torch.int32)
    if bool((valid < 2).any()) or bool((valid > actual_vertices).any()):
        raise ValueError(
            "Query-Robust num_valid_vertices must be in [2, M] for every "
            f"(layer, kv_head), got min={int(valid.min())} max={int(valid.max())} M={actual_vertices}."
        )
    for layer in range(actual_layers):
        for kv_head in range(actual_global_heads):
            valid_count = int(valid[layer, kv_head])
            if valid_count == actual_vertices:
                continue
            real_vertices = vertices[layer, kv_head, :valid_count]
            padded_vertices = vertices[layer, kv_head, valid_count:]
            is_repeat = (padded_vertices[:, None, :] == real_vertices[None, :, :]).all(dim=-1).any(dim=-1)
            if not bool(is_repeat.all()):
                raise ValueError(
                    "Query-Robust padded vertices must repeat one of the valid "
                    f"support vertices at layer={layer}, kv_head={kv_head}."
                )
    local_heads = actual_global_heads // tp_size
    head_start = rank * local_heads
    head_end = head_start + local_heads
    sliced_vertices = vertices[:, head_start:head_end].to(torch.float32).contiguous()
    sliced_valid = valid[:, head_start:head_end].contiguous()
    if tuple(sliced_vertices.shape[:2]) != (int(num_layers), local_heads):
        raise RuntimeError("Query-Robust TP slicing produced an invalid local shape.")
    if device is not None:
        sliced_vertices = sliced_vertices.to(device=device)
        sliced_valid = sliced_valid.to(device=device)
    return QueryRobustAsset(
        vertices=sliced_vertices,
        num_valid_vertices=sliced_valid,
        meta=dict(meta),
    )


def _validate_solver_inputs(keys: torch.Tensor, vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if keys.ndim != 2 or vertices.ndim != 2:
        raise ValueError(
            "Query-Robust solver expects keys [N, D] and vertices [M, D], got "
            f"keys={tuple(keys.shape)} vertices={tuple(vertices.shape)}."
        )
    if int(keys.shape[0]) <= 0 or int(vertices.shape[0]) < 2:
        raise ValueError("Query-Robust pages need N > 0 and at least two vertices.")
    if keys.shape[1] != vertices.shape[1]:
        raise ValueError("Query-Robust keys and vertices must share head_dim.")
    return keys.float(), vertices.float()


@torch.no_grad()
def solve_query_robust_page(
    keys: torch.Tensor,
    vertices: torch.Tensor,
    *,
    scale: float,
    solver_iters: int = 24,
    solver_lr: float = 0.25,
    uniform_p: bool = False,
) -> QueryRobustSummary:
    """Solve one page's minimax dual in stable FP32 arithmetic.

    The exact page mass is logsumexp(s q^T k). For any distribution p over the
    page keys, the Gibbs variational principle gives the affine lower bound
    s q^T sum(p k) + H(p). The slack is a KL divergence, and its convexity
    means the maximum over a query hull is attained at a vertex.
    """

    keys, vertices = _validate_solver_inputs(keys, vertices)
    if not math.isfinite(float(scale)):
        raise ValueError(f"Query-Robust attention scale must be finite, got {scale}.")
    solver_iters = int(solver_iters)
    solver_lr = float(solver_lr)
    if solver_iters <= 0 or not math.isfinite(solver_lr) or solver_lr <= 0:
        raise ValueError("Query-Robust solver_iters and solver_lr must be positive.")

    logits = float(scale) * (vertices @ keys.transpose(0, 1))
    page_lse = torch.logsumexp(logits, dim=-1)
    lambda_logits = torch.zeros(int(vertices.shape[0]), dtype=torch.float32, device=keys.device)
    if uniform_p:
        p = torch.full((int(keys.shape[0]),), 1.0 / float(keys.shape[0]), dtype=torch.float32, device=keys.device)
        lam = torch.full_like(lambda_logits, 1.0 / float(vertices.shape[0]))
        bar_s = lam @ logits
    else:
        for step_index in range(solver_iters):
            lam = torch.softmax(lambda_logits, dim=0)
            bar_s = lam @ logits
            p = torch.softmax(bar_s, dim=0)
            grad = page_lse - logits @ p
            grad = grad - grad.mean()
            grad_scale = grad.abs().max().clamp_min(1e-6)
            step = solver_lr / grad_scale / math.sqrt(1.0 + float(step_index))
            lambda_logits = lambda_logits + step * grad
        # Recompute p after the final lambda update; the in-loop p is stale.
        lam = torch.softmax(lambda_logits, dim=0)
        bar_s = lam @ logits
        p = torch.softmax(bar_s, dim=0)

    landmark = p @ keys
    entropy = (
        torch.full((), math.log(float(keys.shape[0])), dtype=torch.float32, device=keys.device)
        if uniform_p
        else torch.logsumexp(bar_s, dim=0) - torch.sum(p * bar_s)
    )
    # The runtime stores the landmark in BF16.  The certificate must describe
    # that stored value, not the higher-precision transient used by the solver.
    stored_landmark = landmark.to(torch.bfloat16).float()
    errors = page_lse - (float(scale) * (vertices @ stored_landmark) + entropy)
    epsilon = errors.max().clamp_min(0.0)
    dual = errors.mean() if uniform_p else torch.sum(lam * errors)
    return QueryRobustSummary(
        landmark=landmark.to(torch.bfloat16),
        bias=entropy.float(),
        epsilon=epsilon.float(),
        dual=dual.float(),
        gap=(epsilon - dual).float(),
        errors=errors.float(),
    )


@torch.no_grad()
def build_query_robust_page_summaries(
    keys: torch.Tensor,
    vertices: torch.Tensor,
    *,
    scale: float,
    solver_iters: int,
    solver_lr: float,
    uniform_p: bool,
) -> QueryRobustSummary:
    """Build summaries for keys shaped [pages, page_size, kv_heads, dim]."""

    if keys.ndim != 4 or vertices.ndim != 3:
        raise ValueError(
            "Query-Robust batched builder expects keys [P, N, H, D] and vertices [H, M, D]."
        )
    pages, page_size, num_heads, head_dim = map(int, keys.shape)
    if tuple(vertices.shape[::2]) != (num_heads, head_dim):
        raise ValueError(
            "Query-Robust vertex head dimensions do not match keys: "
            f"keys={tuple(keys.shape)} vertices={tuple(vertices.shape)}."
        )
    flat_keys = keys.float().permute(0, 2, 1, 3).reshape(pages * num_heads, page_size, head_dim)
    flat_vertices = vertices.float().unsqueeze(0).expand(pages, -1, -1, -1).reshape(
        pages * num_heads, int(vertices.shape[1]), head_dim
    )
    logits = float(scale) * torch.bmm(flat_vertices, flat_keys.transpose(1, 2))
    page_lse = torch.logsumexp(logits, dim=-1)
    lambda_logits = torch.zeros(
        pages * num_heads, int(vertices.shape[1]), dtype=torch.float32, device=keys.device
    )
    if uniform_p:
        p = torch.full(
            (pages * num_heads, page_size), 1.0 / float(page_size),
            dtype=torch.float32, device=keys.device,
        )
        lam = torch.full_like(lambda_logits, 1.0 / float(vertices.shape[1]))
        bar_s = torch.bmm(lam.unsqueeze(1), logits).squeeze(1)
    else:
        for step_index in range(int(solver_iters)):
            lam = torch.softmax(lambda_logits, dim=-1)
            bar_s = torch.bmm(lam.unsqueeze(1), logits).squeeze(1)
            p = torch.softmax(bar_s, dim=-1)
            grad = page_lse - torch.bmm(logits, p.unsqueeze(-1)).squeeze(-1)
            grad = grad - grad.mean(dim=-1, keepdim=True)
            grad_scale = grad.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            step = float(solver_lr) / grad_scale / math.sqrt(1.0 + step_index)
            lambda_logits = lambda_logits + step * grad
        lam = torch.softmax(lambda_logits, dim=-1)
        bar_s = torch.bmm(lam.unsqueeze(1), logits).squeeze(1)
        p = torch.softmax(bar_s, dim=-1)

    flat_landmark = torch.bmm(p.unsqueeze(1), flat_keys).squeeze(1)
    entropy = (
        torch.full(
            (pages * num_heads,),
            math.log(float(page_size)),
            dtype=torch.float32,
            device=keys.device,
        )
        if uniform_p
        else torch.logsumexp(bar_s, dim=-1) - torch.sum(p * bar_s, dim=-1)
    )
    stored_landmark = flat_landmark.to(torch.bfloat16).float()
    errors = page_lse - (
        float(scale) * torch.bmm(flat_vertices, stored_landmark.unsqueeze(-1)).squeeze(-1)
        + entropy[:, None]
    )
    epsilon = errors.amax(dim=-1).clamp_min(0.0)
    dual = errors.mean(dim=-1) if uniform_p else torch.sum(lam * errors, dim=-1)
    shape_phd = (pages, num_heads, head_dim)
    shape_ph = (pages, num_heads)
    return QueryRobustSummary(
        landmark=flat_landmark.to(torch.bfloat16).view(shape_phd),
        bias=entropy.float().view(shape_ph),
        epsilon=epsilon.float().view(shape_ph),
        dual=dual.float().view(shape_ph),
        gap=(epsilon - dual).float().view(shape_ph),
        errors=errors.float().view(pages, num_heads, int(vertices.shape[1])),
    )


@torch.no_grad()
def score_query_robust_pages_reference(
    query: torch.Tensor,
    landmark: torch.Tensor,
    bias: torch.Tensor,
    epsilon: torch.Tensor,
    metadata_valid: torch.Tensor,
    row_page_slots: torch.Tensor,
    *,
    scale: float,
    alpha: float,
) -> torch.Tensor:
    """Reference QR page scorer with one shared page route per request."""

    if query.ndim != 3 or landmark.ndim != 3:
        raise ValueError("QR scorer expects query [B, QH, D] and landmark [P, H, D].")
    if bias.shape != epsilon.shape or bias.ndim != 2:
        raise ValueError("QR bias and epsilon must both have shape [P, H].")
    if metadata_valid.shape != bias.shape:
        raise ValueError("QR metadata_valid must match [P, H] scalar metadata.")
    batch, query_heads, dim = map(int, query.shape)
    pages, kv_heads, metadata_dim = map(int, landmark.shape)
    if metadata_dim != dim or query_heads % kv_heads:
        raise ValueError("QR query heads must be divisible by local KV heads.")
    if row_page_slots.shape[0] != batch or row_page_slots.dtype != torch.int32:
        raise ValueError("QR row_page_slots must be int32 [batch, pages].")
    if row_page_slots.ndim != 2:
        raise ValueError("QR row_page_slots must have shape [batch, pages].")
    if row_page_slots.device != query.device:
        raise ValueError("QR scorer inputs must share one device.")
    safe_pages = row_page_slots.to(torch.long).clamp(0, pages - 1)
    width = int(row_page_slots.shape[1])
    page_landmark = landmark.index_select(0, safe_pages.reshape(-1)).view(
        batch, width, kv_heads, dim
    )
    page_bias = bias.index_select(0, safe_pages.reshape(-1)).view(batch, width, kv_heads)
    page_epsilon = epsilon.index_select(0, safe_pages.reshape(-1)).view(batch, width, kv_heads)
    page_valid = metadata_valid.index_select(0, safe_pages.reshape(-1)).view(batch, width, kv_heads)
    page_valid = page_valid & row_page_slots.ge(0).unsqueeze(-1)
    group_size = query_heads // kv_heads
    grouped_query = query.float().view(batch, kv_heads, group_size, dim)
    dots = torch.einsum("bhgd,bwhd->bwhg", grouped_query, page_landmark.float())
    dot_max = dots.amax(dim=-1)
    scores = dot_max * float(scale) + page_bias.float() + float(alpha) * page_epsilon.float()
    scores = scores.amax(dim=-1)
    return torch.where(
        ~page_valid.all(dim=-1),
        torch.full_like(scores, float("inf")),
        scores,
    ).float().contiguous()


class QueryRobustCacheManager(QuestCacheManager):
    """Explicit-KV paged cache manager with QR page summaries."""

    def __init__(
        self,
        config: Config,
        parallel_context: ParallelContext,
        *,
        allocation_budget_bytes: int | None = None,
    ):
        if normalize_sparse_method(config.sparse_method) != "query_robust":
            raise ValueError("QueryRobustCacheManager requires sparse_method='query_robust'.")
        if str(config.attention_cache_layout) != "explicit_kv":
            raise NotImplementedError("Query-Robust currently requires explicit KV storage.")
        super().__init__(config, parallel_context, allocation_budget_bytes=allocation_budget_bytes)
        if not isinstance(self.attention_cache_storage, ExplicitKVStorage):
            raise TypeError(
                "Query-Robust requires ExplicitKVStorage, got "
                f"{type(self.attention_cache_storage).__name__}."
            )
        # QuestCacheManager allocates its [min, max] page-summary tensor during
        # construction. Query-Robust replaces that representation, so release
        # the inherited tensor before allocating the QR metadata buffers.
        # Keeping both live temporarily costs several GiB at large capacities.
        if hasattr(self, "metadata_cache"):
            del self.metadata_cache
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        global_heads = int(getattr(self.hf_config, "num_key_value_heads", 0))
        expected_model_id = str(getattr(config, "model", "")).rstrip("/").split("/")[-1]
        asset = load_query_robust_asset(
            config.query_robust_vertices_path,
            num_layers=int(self.num_layers),
            global_num_kv_heads=global_heads,
            head_dim=int(self.head_dim),
            num_vertices=int(config.query_robust_num_vertices),
            tensor_parallel_rank=int(self.tp_rank),
            tensor_parallel_size=int(self.tp_size),
            expected_model_id=expected_model_id or None,
            expected_model_fingerprint=getattr(config, "query_robust_model_fingerprint", None),
            expected_rope_config=getattr(config, "query_robust_rope_config", None),
            device=self.device,
        )
        if int(asset.vertices.shape[1]) != int(self.num_kv_heads):
            raise ValueError(
                "Query-Robust asset local KV head count does not match runtime: "
                f"asset={int(asset.vertices.shape[1])} runtime={self.num_kv_heads}."
            )
        self.query_robust_vertices = asset.vertices
        self.query_robust_num_valid_vertices = asset.num_valid_vertices
        self.query_robust_asset_meta = asset.meta
        self.query_robust_scale = float(self.head_dim) ** -0.5
        self.query_robust_alpha = float(config.query_robust_score_alpha)
        self._allocate_qr_metadata()
        self.quest_page_selector = resolve_quest_page_selection_provider(
            QuestPageSelectionOpSpec(score_dtype=torch.float32, cuda_graph=bool(config.decode_graph)),
            device_index=self.device.index or 0,
        )

    def _metadata_bytes_per_page_per_layer(self) -> int:
        return int(
            self.num_kv_heads * self.head_dim * torch.tensor([], dtype=torch.bfloat16).element_size()
            + self.num_kv_heads * 2 * torch.tensor([], dtype=torch.float32).element_size()
            + self.num_kv_heads
            + self.num_kv_heads * torch.tensor([], dtype=torch.float32).element_size()
        )

    def _allocate_qr_metadata(self) -> None:
        shape = (int(self.num_kv_layers), int(self.num_pages), int(self.num_kv_heads))
        self.landmark_cache = torch.zeros(
            (*shape, int(self.head_dim)), dtype=torch.bfloat16, device=self.device
        )
        self.metadata_cache = self.landmark_cache
        self.bias_cache = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.epsilon_cache = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.metadata_valid = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.duality_gap_cache = torch.zeros(shape, dtype=torch.float32, device=self.device)

    def _invalidate_pages(self, slots: torch.Tensor) -> None:
        if not hasattr(self, "metadata_valid") or slots.numel() == 0:
            return
        pages = torch.div(
            slots.to(device=self.device, dtype=torch.long),
            int(self.page_size),
            rounding_mode="floor",
        ).clamp(0, int(self.num_pages) - 1)
        pages = torch.unique(pages)
        self.metadata_valid.index_fill_(1, pages, False)
        self.bias_cache.index_fill_(1, pages, 0)
        self.epsilon_cache.index_fill_(1, pages, 0)
        self.duality_gap_cache.index_fill_(1, pages, 0)
        self.landmark_cache.index_fill_(1, pages, 0)

    @torch.no_grad()
    def _allocate(self, seq_id: int, size: int) -> torch.Tensor:
        slots = super()._allocate(seq_id, size)
        self._invalidate_pages(slots)
        return slots

    @torch.no_grad()
    def _allocate_batch(
        self,
        seq_ids: list[int],
        size: int,
        *,
        graph_batch_size: int | None = None,
    ) -> torch.Tensor:
        slots = super()._allocate_batch(seq_ids, size, graph_batch_size=graph_batch_size)
        self._invalidate_pages(slots)
        return slots

    def store_attention_payload(self, layer_idx: int, payload: AttentionCacheWrite) -> torch.Tensor:
        if not isinstance(self.attention_cache_storage, ExplicitKVStorage):
            raise TypeError("Query-Robust only supports ExplicitKVStorage.")
        if not isinstance(payload, ExplicitKVWrite):
            raise TypeError(
                "Query-Robust explicit KV store requires ExplicitKVWrite, got "
                f"{type(payload).__name__}."
            )
        slot_mapping = self.layer_batch_state.slot_mapping
        if slot_mapping is None:
            raise RuntimeError(
                f"Query-Robust KV store requires slot_mapping at layer={layer_idx}."
            )
        self.attention_cache_storage.store(self.kv_layer_index(layer_idx), slot_mapping, payload)
        return slot_mapping

    @torch.no_grad()
    def on_kv_stored(self, layer_idx: int, k: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        if slot_mapping is None or slot_mapping.numel() == 0:
            return
        page_size = int(self.page_size)
        page_slots = torch.div(slot_mapping, page_size, rounding_mode="floor")
        page_offsets = torch.remainder(slot_mapping, page_size)
        sealed = slot_mapping.ge(0) & page_offsets.eq(page_size - 1)
        is_prefill = bool(get_context().is_prefill)
        is_capturing = self._is_stream_capturing()
        # Decode always uses fixed-size candidate tensors.  This avoids a
        # device-to-host reduction in the graph and eager decode paths.
        if is_prefill and not is_capturing and not bool(sealed.any()):
            return
        kv_idx = self.kv_layer_index(layer_idx)
        k_cache = self.attention_cache_storage.layer_payload(kv_idx).k_cache
        if is_capturing or not is_prefill:
            candidate_pages = page_slots.clamp(0, int(self.num_pages) - 1).to(torch.long)
            active = sealed
        else:
            candidate_pages = page_slots[sealed].to(torch.long)
            active = torch.ones(
                int(candidate_pages.numel()), dtype=torch.bool, device=candidate_pages.device
            )
        if candidate_pages.numel() == 0:
            return
        vertices = self.query_robust_vertices[int(layer_idx)]
        with profiler.record("query_robust_build_summary"):
            if k_cache.is_cuda:
                from sparsevllm.kernels.triton.query_robust_summary import (
                    build_query_robust_page_summaries_from_cache,
                )

                (
                    landmark,
                    bias,
                    epsilon,
                    dual,
                    gap,
                    errors,
                ) = build_query_robust_page_summaries_from_cache(
                    k_cache,
                    candidate_pages.to(torch.int32).contiguous(),
                    vertices.contiguous(),
                    active.contiguous(),
                    page_size=page_size,
                    scale=self.query_robust_scale,
                    solver_iters=int(self.config.query_robust_solver_iters),
                    solver_lr=float(self.config.query_robust_solver_lr),
                    uniform_p=bool(self.config.query_robust_uniform_p),
                )
                summary = QueryRobustSummary(
                    landmark=landmark,
                    bias=bias,
                    epsilon=epsilon,
                    dual=dual,
                    gap=gap,
                    errors=errors,
                )
            else:
                token_slots = (
                    candidate_pages[:, None] * page_size
                    + torch.arange(page_size, device=self.device, dtype=torch.long)[None, :]
                )
                keys = k_cache.index_select(0, token_slots.reshape(-1)).view(
                    int(candidate_pages.numel()),
                    page_size,
                    int(self.num_kv_heads),
                    int(self.head_dim),
                )
                summary = build_query_robust_page_summaries(
                    keys,
                    vertices,
                    scale=self.query_robust_scale,
                    solver_iters=int(self.config.query_robust_solver_iters),
                    solver_lr=float(self.config.query_robust_solver_lr),
                    uniform_p=bool(self.config.query_robust_uniform_p),
                )
        old_landmark = self.landmark_cache[kv_idx].index_select(0, candidate_pages)
        old_bias = self.bias_cache[kv_idx].index_select(0, candidate_pages)
        old_epsilon = self.epsilon_cache[kv_idx].index_select(0, candidate_pages)
        old_gap = self.duality_gap_cache[kv_idx].index_select(0, candidate_pages)
        old_valid = self.metadata_valid[kv_idx].index_select(0, candidate_pages)
        mask = active[:, None]
        self.landmark_cache[kv_idx].index_copy_(
            0, candidate_pages, torch.where(mask[..., None], summary.landmark, old_landmark)
        )
        self.bias_cache[kv_idx].index_copy_(
            0, candidate_pages, torch.where(mask, summary.bias, old_bias)
        )
        self.epsilon_cache[kv_idx].index_copy_(
            0, candidate_pages, torch.where(mask, summary.epsilon, old_epsilon)
        )
        self.duality_gap_cache[kv_idx].index_copy_(
            0, candidate_pages, torch.where(mask, summary.gap, old_gap)
        )
        self.metadata_valid[kv_idx].index_copy_(
            0, candidate_pages, torch.where(mask, True, old_valid)
        )

    def free_seq(self, seq_id: int):
        row_idx = self.seq_id_to_row.get(int(seq_id))
        pages_to_invalidate: list[int] = []
        if row_idx is not None:
            row_len = int(self.row_seq_lens[row_idx])
            cached_pages = self.seq_id_to_cached_pages.get(int(seq_id), set())
            for page_index in range((row_len + self.page_size - 1) // self.page_size):
                if page_index not in cached_pages:
                    page_slot = int(self.buffer_req_to_page_slots_cpu[row_idx, page_index])
                    if page_slot >= 0:
                        pages_to_invalidate.append(page_slot * self.page_size)
        super().free_seq(seq_id)
        if pages_to_invalidate:
            self._invalidate_pages(
                torch.tensor(pages_to_invalidate, dtype=torch.int32, device=self.device)
            )

    def reset_after_warmup(self) -> None:
        super().reset_after_warmup()
        self.metadata_valid.zero_()
        self.landmark_cache.zero_()
        self.bias_cache.zero_()
        self.epsilon_cache.zero_()
        self.duality_gap_cache.zero_()

    @torch.no_grad()
    def query_robust_duality_gap_stats(self) -> dict[str, float | int]:
        """Return an explicit diagnostic snapshot of solved-page gaps."""

        gaps = self.duality_gap_cache[self.metadata_valid].float()
        if gaps.numel() == 0:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        quantiles = torch.quantile(
            gaps,
            torch.tensor((0.50, 0.95), dtype=torch.float32, device=gaps.device),
        )
        return {
            "count": int(gaps.numel()),
            "p50": float(quantiles[0].item()),
            "p95": float(quantiles[1].item()),
            "max": float(gaps.max().item()),
        }

    @torch.no_grad()
    def _score_previous_decode_pages(
        self,
        layer_idx: int,
        score_query: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        *,
        max_pages: int,
        num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(num_kv_heads) != int(self.num_kv_heads):
            raise ValueError(
                "Query-Robust attention/metadata head contract is inconsistent: "
                f"attention_kv_heads={num_kv_heads} metadata_heads={self.num_kv_heads}."
            )
        batch_size = int(score_query.shape[0])
        num_pages = getattr(self, "_decode_num_pages", None)
        previous_page_counts = getattr(self, "_decode_previous_page_counts", None)
        if (
            num_pages is None
            or previous_page_counts is None
            or getattr(self, "_decode_page_geometry_context_lens", None) is not context_lens
            or int(num_pages.numel()) != batch_size
        ):
            num_pages = torch.empty_like(context_lens)
            previous_page_counts = torch.empty_like(context_lens)
            self._decode_num_pages = num_pages
            self._decode_previous_page_counts = previous_page_counts
            self._prepare_decode_page_geometry(context_lens)
        elif not getattr(self, "_decode_page_geometry_ready", False):
            self._prepare_decode_page_geometry(context_lens)
        row_page_slots = getattr(self, "_decode_row_page_slots", None)
        if (
            row_page_slots is None
            or getattr(self, "_decode_row_page_slots_req_indices", None) is not req_indices
            or int(row_page_slots.shape[0]) != batch_size
            or int(row_page_slots.shape[1]) < int(max_pages)
        ):
            row_page_slots = self.buffer_req_to_page_slots.index_select(0, req_indices)[:, : int(max_pages)]
        else:
            row_page_slots = row_page_slots[:, : int(max_pages)]
        kv_idx = self.kv_layer_index(layer_idx)
        if score_query.is_cuda:
            from sparsevllm.kernels.triton.query_robust import score_query_robust_pages

            page_scores = score_query_robust_pages(
                score_query.contiguous(),
                self.landmark_cache[kv_idx],
                self.bias_cache[kv_idx],
                self.epsilon_cache[kv_idx],
                self.metadata_valid[kv_idx],
                row_page_slots.contiguous(),
                scale=self.query_robust_scale,
                alpha=self.query_robust_alpha,
            )
        else:
            page_scores = score_query_robust_pages_reference(
                score_query.contiguous(),
                self.landmark_cache[kv_idx],
                self.bias_cache[kv_idx],
                self.epsilon_cache[kv_idx],
                self.metadata_valid[kv_idx],
                row_page_slots.contiguous(),
                scale=self.query_robust_scale,
                alpha=self.query_robust_alpha,
            )
        return page_scores.contiguous(), row_page_slots.contiguous(), num_pages, previous_page_counts


__all__ = [
    "QueryRobustAsset",
    "QueryRobustCacheManager",
    "QueryRobustSummary",
    "build_query_robust_page_summaries",
    "load_query_robust_asset",
    "score_query_robust_pages_reference",
    "solve_query_robust_page",
]
