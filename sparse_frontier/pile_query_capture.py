"""Capture an architecture-correct empirical query pool from The Pile.

The default command follows the Q-Filters-style calibration budget: 20 raw
Pile sequences, exactly 2,048 tokenizer tokens per sequence, and 3,000
uniform samples without replacement for every layer/query-head pair.  Queries
are captured after any Q normalization and RoPE, before the attention scale;
no SVD or learned projection is applied.

The implementation uses the Hugging Face datasets-server API instead of the
``datasets`` package so that calibration remains reproducible in the small
runtime environment used by this repository.  The selected row IDs, dataset
revision, tokenizer/model revisions, seed, and token positions are written to
``manifest.json`` and each layer shard.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Sequence

import requests
import torch
from torch import Tensor


DEFAULT_DATASET = "monology/pile-uncopyrighted"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "train"
DATASET_SERVER = "https://datasets-server.huggingface.co"


def _safe_load_shard(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"query-pool shard is missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.6 compatibility
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"query-pool shard is corrupt: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"query-pool shard is not a dictionary: {path}")
    return payload


@dataclass(frozen=True)
class PileSequence:
    sequence_index: int
    sequence_id: str
    token_ids: list[int]
    source_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class PileQueryPoolLayer:
    """Layer-local queries grouped by KV head for robust chunk fitting."""

    queries_by_kv_head: Tensor  # [local_kv_heads, group_size*samples, head_dim]
    weights_by_kv_head: Tensor  # [local_kv_heads, group_size*samples]
    positions_by_kv_head: Tensor  # [local_kv_heads, group_size*samples]
    query_head_ids_by_kv_head: Tensor  # [local_kv_heads, group_size*samples]


@dataclass(frozen=True)
class PileQueryPool:
    manifest: dict[str, Any]
    layers: tuple[PileQueryPoolLayer, ...]


def load_pile_query_pool(path: str | Path) -> PileQueryPool:
    """Load a schema-2 per-Q-head empirical pool for robust fitting."""

    root = Path(path)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Pile query-pool manifest: {exc}") from exc
    if manifest.get("artifact_type") not in {
        "pile_empirical_query_pool",
        "vllm_empirical_query_pool",
    } or manifest.get("schema_version") != 2:
        raise ValueError("query_robust requires a schema-2 empirical query pool")
    layers: list[PileQueryPoolLayer] = []
    q_heads = int(manifest["num_q_heads"])
    kv_heads = int(manifest["num_kv_heads"])
    if int(manifest.get("tp_size", 1)) != 1:
        raise ValueError("schema-2 Pile query-pool loading currently requires TP=1")
    samples = int(manifest["sampling"]["samples_per_layer_query_head"])
    head_dim = int(manifest["head_dim"])
    if q_heads % kv_heads:
        raise ValueError("Pile query pool has invalid GQA geometry")
    group = q_heads // kv_heads
    expected_weight = 1.0 / (group * samples)
    for layer_idx in range(int(manifest["num_layers"])):
        shard_path = root / f"layer_{layer_idx:03d}.pt"
        payload = _safe_load_shard(shard_path)
        required = {"queries", "query_head_id", "kv_head_id", "position", "weights", "sequence_index"}
        if set(payload) - (required | {"representation"}) or not required.issubset(payload):
            raise ValueError(f"Pile query shard {shard_path} has invalid fields")
        queries = payload["queries"]
        positions = payload["position"]
        query_ids = payload["query_head_id"]
        kv_ids = payload["kv_head_id"]
        weights = payload["weights"]
        expected = (q_heads, samples)
        if queries.shape != (q_heads, samples, head_dim) or positions.shape != expected:
            raise ValueError(f"Pile query shard {shard_path} has invalid tensor shapes")
        if query_ids.shape != expected or kv_ids.shape != expected or weights.shape != expected:
            raise ValueError(f"Pile query shard {shard_path} has invalid metadata shapes")
        if queries.dtype != torch.bfloat16 or positions.dtype not in {torch.int32, torch.int64}:
            raise ValueError(f"Pile query shard {shard_path} has invalid dtypes")
        if not torch.isfinite(queries).all() or not torch.isfinite(weights).all():
            raise ValueError(f"Pile query shard {shard_path} contains non-finite tensors")
        if torch.any(weights < 0):
            raise ValueError(f"Pile query shard {shard_path} has negative weights")
        torch.testing.assert_close(weights.sum(-1), torch.ones(q_heads), atol=1e-6, rtol=1e-6)
        grouped_q, grouped_w, grouped_pos, grouped_ids = [], [], [], []
        for kv in range(kv_heads):
            heads = torch.arange(kv * group, (kv + 1) * group)
            grouped_q.append(queries[heads].reshape(group * samples, head_dim))
            grouped_w.append(torch.full((group * samples,), expected_weight, dtype=torch.float32))
            grouped_pos.append(positions[heads].reshape(group * samples).to(torch.int32))
            grouped_ids.append(query_ids[heads].reshape(group * samples).to(torch.int16))
        layers.append(
            PileQueryPoolLayer(
                queries_by_kv_head=torch.stack(grouped_q),
                weights_by_kv_head=torch.stack(grouped_w),
                positions_by_kv_head=torch.stack(grouped_pos),
                query_head_ids_by_kv_head=torch.stack(grouped_ids),
            )
        )
    return PileQueryPool(manifest=manifest, layers=tuple(layers))


def finalize_vllm_capture_query_pool(
    input_dir: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    *,
    samples_per_head: int,
    seed: int,
    tasks: Sequence[str] | None = None,
) -> Path:
    """Freeze dense vLLM calibration captures as a transportable schema-2 pool.

    The raw collector saves each decode query at the *origin* RoPE position
    (its decode offset) and verifies composition against the backend's actual
    post-RoPE query.  Retaining that origin plus its offset lets the router
    transport the frozen query to the current prompt's routing horizon without
    needing task text or a model forward in the online attention path.
    """

    selected_tasks = None if tasks is None else {str(task) for task in tasks}
    if selected_tasks is not None and (not selected_tasks or any(not task for task in selected_tasks)):
        raise ValueError("task filter must contain one or more nonempty task names")
    sources = (
        [Path(item) for item in input_dir]
        if not isinstance(input_dir, (str, Path))
        else [Path(input_dir)]
    )
    if not sources:
        raise ValueError("at least one raw capture directory is required")
    output = Path(output_dir)
    if samples_per_head < 1:
        raise ValueError("samples_per_head must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    try:
        raw_manifest = json.loads((sources[0] / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read raw capture manifest: {exc}") from exc
    required_manifest = {
        "model_id", "model_revision", "num_layers", "num_q_heads", "num_kv_heads",
        "tp_size", "head_dim", "attention_scale", "rope_type", "rope_parameters",
        "representation",
    }
    missing = sorted(required_manifest - set(raw_manifest))
    if missing:
        raise ValueError(f"raw capture manifest is missing {missing}")
    num_layers = int(raw_manifest["num_layers"])
    num_q_heads = int(raw_manifest["num_q_heads"])
    num_kv_heads = int(raw_manifest["num_kv_heads"])
    head_dim = int(raw_manifest["head_dim"])
    if int(raw_manifest["tp_size"]) != 1 or num_q_heads % num_kv_heads:
        raise ValueError("schema-2 vLLM capture finalization currently requires TP=1 with valid GQA")
    # Independent dense capture invocations are intentionally written to
    # separate directories so request-local step names cannot collide.  Their
    # manifests must agree on every representation-defining field before their
    # calibration rows are combined into one deterministic empirical pool.
    identity_fields = required_manifest
    for source in sources[1:]:
        try:
            other = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read raw capture manifest {source}: {exc}") from exc
        mismatches = [
            name for name in identity_fields
            if other.get(name) != raw_manifest.get(name)
        ]
        if mismatches:
            raise ValueError(
                "raw capture manifests are incompatible: " + ", ".join(sorted(mismatches))
            )

    candidates: list[list[list[tuple[Tensor, int, int]]]] = [
        [[] for _ in range(num_q_heads)] for _ in range(num_layers)
    ]
    sequence_ids: set[int] = set()
    steps = sorted(
        step for source in sources for step in source.rglob("step*.pt")
    )
    if not steps:
        raise ValueError("raw capture contains no decode-query step shards")
    for step in steps:
        payload = _safe_load_shard(step)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("split") != "calibration":
            continue
        if selected_tasks is not None:
            task = metadata.get("task")
            if not isinstance(task, str):
                raise ValueError(f"raw capture shard {step} has no task label for task filtering")
            if task not in selected_tasks:
                continue
        layer_ids = payload.get("layer_ids")
        queries = payload.get("prompt_origin_queries")
        if not isinstance(layer_ids, Tensor) or layer_ids.ndim != 1:
            raise ValueError(f"raw capture shard {step} has invalid layer IDs")
        if not isinstance(queries, Tensor) or queries.shape != (
            layer_ids.numel(), num_q_heads, head_dim
        ):
            raise ValueError(f"raw capture shard {step} has invalid prompt-origin queries")
        position = int(metadata.get("decode_offset", -1))
        sequence_id = int(metadata.get("sequence_id_hash", -1))
        if position < 0 or sequence_id < 0:
            raise ValueError(f"raw capture shard {step} has invalid decode metadata")
        sequence_ids.add(sequence_id)
        for local_layer, layer_idx in enumerate(layer_ids.tolist()):
            if not 0 <= int(layer_idx) < num_layers:
                raise ValueError(f"raw capture shard {step} has out-of-range layer ID")
            for q_head in range(num_q_heads):
                candidates[int(layer_idx)][q_head].append(
                    (queries[local_layer, q_head].to(torch.bfloat16).cpu(), position, sequence_id)
                )
    if not sequence_ids:
        raise ValueError("raw capture contains no calibration split queries")
    if any(len(candidates[layer][head]) < samples_per_head for layer in range(num_layers) for head in range(num_q_heads)):
        raise ValueError("raw calibration capture does not contain enough queries per layer/head")

    output.mkdir(parents=True, exist_ok=True)
    group_size = num_q_heads // num_kv_heads
    sequence_index = {value: index for index, value in enumerate(sorted(sequence_ids))}
    for layer_idx in range(num_layers):
        query_heads: list[Tensor] = []
        positions: list[Tensor] = []
        sequences: list[Tensor] = []
        for q_head in range(num_q_heads):
            rows = candidates[layer_idx][q_head]
            generator = torch.Generator().manual_seed(seed + layer_idx * 1009 + q_head)
            chosen = torch.randperm(len(rows), generator=generator)[:samples_per_head].tolist()
            query_heads.append(torch.stack([rows[index][0] for index in chosen]))
            positions.append(torch.tensor([rows[index][1] for index in chosen], dtype=torch.int32))
            sequences.append(torch.tensor([sequence_index[rows[index][2]] for index in chosen], dtype=torch.int32))
        q_ids = torch.arange(num_q_heads, dtype=torch.int16)[:, None].expand(num_q_heads, samples_per_head)
        _atomic_torch_save(
            {
                "queries": torch.stack(query_heads),
                "query_head_id": q_ids.clone(),
                "kv_head_id": (q_ids // group_size).clone(),
                "position": torch.stack(positions),
                "sequence_index": torch.stack(sequences),
                "weights": torch.full((num_q_heads, samples_per_head), 1.0 / samples_per_head, dtype=torch.float32),
                "representation": raw_manifest["representation"],
            },
            output / f"layer_{layer_idx:03d}.pt",
        )
    manifest = {
        "schema_version": 2,
        "artifact_type": "vllm_empirical_query_pool",
        "model_id": raw_manifest["model_id"],
        "model_revision": raw_manifest["model_revision"],
        "tokenizer_revision": raw_manifest["model_revision"],
        "num_layers": num_layers,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "tp_size": 1,
        "head_dim": head_dim,
        "gqa_group_size": group_size,
        "attention_scale": float(raw_manifest["attention_scale"]),
        "rope_type": raw_manifest["rope_type"],
        "rope_parameters": raw_manifest["rope_parameters"],
        "representation": raw_manifest["representation"],
        "normalization": {"q_norm": True, "k_norm": True},
        "dataset": {
            "source": "dense_vllm_calibration_capture",
            "input": [str(source) for source in sources],
        },
        "sampling": {
            "method": "per_layer_query_head_uniform_without_replacement",
            "without_replacement": True,
            "samples_per_layer_query_head": samples_per_head,
            "seed": seed,
            "task_filter": sorted(selected_tasks) if selected_tasks is not None else None,
        },
        "sequence_records": [{"sequence_index": value} for value in sorted(sequence_ids)],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _get_json(url: str, *, timeout: float = 60.0) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after is not None else min(30.0, 2.0 ** attempt)
                time.sleep(max(0.5, delay))
                response.raise_for_status()
            response.raise_for_status()
            payload = response.json()
            break
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == 5:
                raise RuntimeError(f"Hugging Face dataset request failed: {url}: {exc}") from exc
            time.sleep(min(30.0, 2.0 ** attempt))
    else:  # pragma: no cover - defensive for unusual requests behavior
        raise RuntimeError(f"Hugging Face dataset request failed: {url}: {last_error}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Hugging Face dataset response is not an object: {url}")
    return payload


def dataset_revision(dataset: str) -> str:
    # The Hub API treats the repository owner/name slash as a path separator.
    encoded = dataset
    payload = _get_json(f"https://huggingface.co/api/datasets/{encoded}")
    revision = payload.get("sha")
    if not isinstance(revision, str) or not revision:
        raise RuntimeError(f"dataset API did not return an immutable revision for {dataset}")
    return revision


def _fetch_row(dataset: str, config: str, split: str, row_index: int) -> dict[str, Any]:
    from urllib.parse import quote

    query = (
        f"{DATASET_SERVER}/rows?dataset={quote(dataset, safe='')}&config={quote(config)}"
        f"&split={quote(split)}&offset={int(row_index)}&length=1"
    )
    payload = _get_json(query)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"dataset row {row_index} is unavailable")
    row = rows[0]
    if not isinstance(row, dict) or not isinstance(row.get("row"), dict):
        raise RuntimeError(f"dataset row {row_index} has an invalid response shape")
    value = row["row"]
    text = value.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"dataset row {row_index} does not contain text")
    metadata = value.get("meta") if isinstance(value.get("meta"), dict) else {}
    return {
        "row_idx": int(row.get("row_idx", row_index)),
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "pile_set_name": metadata.get("pile_set_name"),
    }


def _fetch_rows(
    dataset: str, config: str, split: str, offset: int, length: int
) -> list[dict[str, Any]]:
    """Fetch one bounded page; batching avoids dataset-server rate limits."""

    from urllib.parse import quote

    query = (
        f"{DATASET_SERVER}/rows?dataset={quote(dataset, safe='')}&config={quote(config)}"
        f"&split={quote(split)}&offset={int(offset)}&length={int(length)}"
    )
    payload = _get_json(query)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"dataset page {offset}:{length} has an invalid response shape")
    result: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict) or not isinstance(raw.get("row"), dict):
            continue
        value = raw["row"]
        text = value.get("text")
        if not isinstance(text, str):
            continue
        metadata = value.get("meta") if isinstance(value.get("meta"), dict) else {}
        result.append(
            {
                "row_idx": int(raw.get("row_idx", offset + len(result))),
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "pile_set_name": metadata.get("pile_set_name"),
            }
        )
    if not result:
        raise RuntimeError(f"dataset page {offset}:{length} contains no usable text rows")
    return result


def select_pile_sequences(
    tokenizer: Any,
    *,
    num_sequences: int = 20,
    sequence_tokens: int = 2048,
    seed: int = 43,
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
) -> tuple[list[PileSequence], dict[str, Any]]:
    """Fetch deterministic raw Pile rows and tokenize exact-length sequences."""

    if num_sequences < 1 or sequence_tokens < 1:
        raise ValueError("num_sequences and sequence_tokens must be positive")
    size_query = (
        f"{DATASET_SERVER}/size?dataset={dataset.replace('/', '%2F')}"
        f"&config={config}&split={split}"
    )
    size_payload = _get_json(size_query)
    total_rows = int(
        size_payload.get("size", {}).get("splits", [{}])[0].get("num_rows", 0)
        or size_payload.get("size", {}).get("dataset", {}).get("num_rows_memory", 0)
    )
    # The API's estimated row count is not a valid offset bound.  Requesting
    # a conservative random row range works for both current Pile mirrors.
    if total_rows < 1:
        total_rows = 891_748 if dataset == DEFAULT_DATASET else 100_000
    rng = random.Random(seed)
    # Select the rows from one reproducible bounded page.  This is still a
    # random document sample, but requires one API request instead of twenty
    # independent requests that trigger the public service rate limiter.
    page_length = min(100, total_rows)
    page_offset = rng.randrange(max(1, total_rows - page_length + 1))
    page = _fetch_rows(dataset, config, split, page_offset, page_length)
    page_by_index = {int(row["row_idx"]): row for row in page}
    selected_indices = rng.sample(range(len(page)), k=min(len(page), num_sequences))
    offsets = [int(page[index]["row_idx"]) for index in selected_indices]
    sequences: list[PileSequence] = []
    source_manifest: list[dict[str, Any]] = []
    for sequence_index, offset in enumerate(offsets):
        token_ids: list[int] = []
        source_rows: list[dict[str, Any]] = []
        row_index = int(offset)
        attempts = 0
        while len(token_ids) < sequence_tokens:
            if row_index in page_by_index:
                row = page_by_index[row_index]
            else:
                row = _fetch_row(dataset, config, split, row_index)
            encoded = tokenizer(
                row["text"],
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            encoded = [int(token) for token in encoded]
            remaining = sequence_tokens - len(token_ids)
            token_ids.extend(encoded[:remaining])
            source_rows.append(
                {
                    "row_idx": row["row_idx"],
                    "text_sha256": row["text_sha256"],
                    "pile_set_name": row["pile_set_name"],
                    "tokens_consumed": min(len(encoded), remaining),
                }
            )
            row_index += 1
            attempts += 1
            if attempts > 32 and len(token_ids) < sequence_tokens:
                raise RuntimeError(
                    f"could not construct {sequence_tokens} tokens near Pile row {offset}"
                )
        identity = json.dumps(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "seed": seed,
                "sequence_index": sequence_index,
                "source_rows": source_rows,
                "token_ids": token_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        sequence_id = hashlib.sha256(identity).hexdigest()[:24]
        sequences.append(PileSequence(sequence_index, sequence_id, token_ids, source_rows))
        source_manifest.append(
            {
                "sequence_index": sequence_index,
                "sequence_id": sequence_id,
                "source_rows": source_rows,
                "token_count": len(token_ids),
            }
        )
    return sequences, {
        "dataset": dataset,
        "config": config,
        "split": split,
        "dataset_revision": dataset_revision(dataset),
        "seed": seed,
        "num_sequences": num_sequences,
        "sequence_tokens": sequence_tokens,
        "source_sequences": source_manifest,
    }


class QueryReservoir:
    """Per-layer/head priority reservoir with exact uniform sampling."""

    def __init__(self, num_layers: int, num_q_heads: int, head_dim: int, size: int, seed: int):
        if min(num_layers, num_q_heads, head_dim, size) < 1:
            raise ValueError("reservoir dimensions must be positive")
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
        self.position = torch.full(
            (num_layers, num_q_heads, size), -1, dtype=torch.int32
        )
        self.sequence_index = torch.full(
            (num_layers, num_q_heads, size), -1, dtype=torch.int32
        )

    @torch.no_grad()
    def update(self, layer: int, queries: Tensor, positions: Tensor, sequence_index: int) -> None:
        """Merge one [heads,tokens,dimension] post-RoPE batch by random priority."""

        if queries.ndim != 3 or queries.shape[0] != self.num_q_heads or queries.shape[2] != self.head_dim:
            raise ValueError("reservoir query batch must have shape [q_heads,tokens,head_dim]")
        if positions.ndim != 1 or positions.numel() != queries.shape[1]:
            raise ValueError("reservoir positions do not match query tokens")
        batch = queries.detach().to(device="cpu", dtype=torch.bfloat16)
        batch_priority = torch.rand(
            self.num_q_heads, queries.shape[1], generator=self.generator, dtype=torch.float32
        )
        for head in range(self.num_q_heads):
            candidate_priority = torch.cat([self.priority[layer, head], batch_priority[head]])
            candidate_queries = torch.cat([self.queries[layer, head], batch[head]], dim=0)
            candidate_positions = torch.cat(
                [self.position[layer, head], positions.to(torch.int32).cpu()], dim=0
            )
            candidate_sequences = torch.cat(
                [self.sequence_index[layer, head], torch.full((queries.shape[1],), sequence_index, dtype=torch.int32)]
            )
            selected_priority, selected = torch.topk(
                candidate_priority, k=self.size, largest=True, sorted=True
            )
            self.priority[layer, head] = selected_priority
            self.queries[layer, head] = candidate_queries[selected]
            self.position[layer, head] = candidate_positions[selected]
            self.sequence_index[layer, head] = candidate_sequences[selected]

    def complete(self) -> None:
        if torch.any(self.position < 0):
            raise RuntimeError("query reservoir did not receive enough tokens")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.no_grad()
def capture_pile_query_pool(
    model_path: str | Path,
    output_dir: str | Path,
    *,
    num_sequences: int = 20,
    sequence_tokens: int = 2048,
    samples_per_head: int = 3000,
    seed: int = 43,
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    device: str | None = None,
) -> Path:
    """Run a frozen Transformers model and write the Pile query artifact."""

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    model_root = Path(model_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_root, use_fast=True)
    sequences, source = select_pile_sequences(
        tokenizer,
        num_sequences=num_sequences,
        sequence_tokens=sequence_tokens,
        seed=seed,
        dataset=dataset,
        config=config,
        split=split,
    )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if device is not None:
        model_kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(model_root, **model_kwargs)
    model.eval()
    config_obj = model.config
    num_layers = int(config_obj.num_hidden_layers)
    num_q_heads = int(config_obj.num_attention_heads)
    num_kv_heads = int(getattr(config_obj, "num_key_value_heads", num_q_heads))
    head_dim = int(getattr(config_obj, "head_dim", config_obj.hidden_size // num_q_heads))
    if num_q_heads % num_kv_heads:
        raise ValueError("model query heads must be divisible by KV heads")
    first_parameter = next(model.parameters())
    input_device = first_parameter.device if device is None or device == "auto" else torch.device(device)
    reservoir = QueryReservoir(num_layers, num_q_heads, head_dim, samples_per_head, seed)
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        attention = layer.self_attn

        def capture_hook(module, args, kwargs, *, _layer=layer_idx):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                raise RuntimeError("attention pre-hook did not receive hidden states")
            projected = module.q_proj(hidden)
            batch, tokens, _ = projected.shape
            q = projected.view(batch, tokens, num_q_heads, head_dim).transpose(1, 2)
            q_norm = getattr(module, "q_norm", None)
            if q_norm is not None:
                q = q_norm(q)
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None:
                raise RuntimeError("model attention did not expose position embeddings")
            cos, sin = position_embeddings
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                positions = torch.arange(tokens, device=q.device, dtype=torch.int32)
            else:
                positions = position_ids.reshape(-1).to(device=q.device, dtype=torch.int32)
            if batch != 1:
                raise RuntimeError("Pile calibration expects one sequence per forward")
            reservoir.update(_layer, q[0], positions, _current_sequence[0])

        hooks.append(attention.register_forward_pre_hook(capture_hook, with_kwargs=True))
    try:
        _current_sequence = [-1]
        for sequence in sequences:
            _current_sequence[0] = sequence.sequence_index
            input_ids = torch.tensor([sequence.token_ids], dtype=torch.long, device=input_device)
            model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    reservoir.complete()

    rope_scaling = dict(getattr(config_obj, "rope_scaling", None) or {})
    rope_type = str(rope_scaling.pop("rope_type", rope_scaling.pop("type", "default")))
    rope_parameters = {**rope_scaling, "rope_theta": getattr(config_obj, "rope_theta", None)}
    model_revision = model_root.name
    manifest = {
        "schema_version": 2,
        "artifact_type": "pile_empirical_query_pool",
        "model_id": str(getattr(config_obj, "_name_or_path", model_root.name)),
        "model_revision": model_revision,
        "tokenizer_revision": model_revision,
        "num_layers": num_layers,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "tp_size": 1,
        "head_dim": head_dim,
        "gqa_group_size": num_q_heads // num_kv_heads,
        "attention_scale": float(head_dim ** -0.5),
        "rope_type": rope_type,
        "rope_parameters": rope_parameters,
        "representation": "post_qk_norm_post_rope_pre_scale",
        "normalization": {
            "q_norm": bool(any(getattr(layer.self_attn, "q_norm", None) is not None for layer in model.model.layers)),
            "k_norm": bool(any(getattr(layer.self_attn, "k_norm", None) is not None for layer in model.model.layers)),
        },
        "dataset": source,
        "sampling": {
            "method": "independent_uniform_priority_reservoir",
            "without_replacement": True,
            "samples_per_layer_query_head": samples_per_head,
            "seed": seed,
        },
        "sequence_records": [asdict(sequence) | {"token_ids": None} for sequence in sequences],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _atomic_torch_save(
        {
            "input_ids": torch.tensor([sequence.token_ids for sequence in sequences], dtype=torch.int32),
            "sequence_ids": [sequence.sequence_id for sequence in sequences],
        },
        output / "calibration_sequences.pt",
    )
    kv_group = num_q_heads // num_kv_heads
    for layer_idx in range(num_layers):
        q_heads = torch.arange(num_q_heads, dtype=torch.int16)
        _atomic_torch_save(
            {
                # Clone each layer slice: saving a view would serialize the
                # entire model-global reservoir backing storage.
                "queries": reservoir.queries[layer_idx].clone(),
                "query_head_id": q_heads[:, None].expand(num_q_heads, samples_per_head).clone(),
                "kv_head_id": (q_heads // kv_group)[:, None].expand(num_q_heads, samples_per_head).clone(),
                "position": reservoir.position[layer_idx].clone(),
                "sequence_index": reservoir.sequence_index[layer_idx].clone(),
                "weights": torch.full(
                    (num_q_heads, samples_per_head),
                    1.0 / samples_per_head,
                    dtype=torch.float32,
                ),
                "representation": manifest["representation"],
            },
            output / f"layer_{layer_idx:03d}.pt",
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_sequences", type=int, default=20)
    parser.add_argument("--sequence_tokens", type=int, default=2048)
    parser.add_argument("--samples_per_head", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    output = capture_pile_query_pool(**vars(args))
    print(json.dumps({"status": "ok", "pool": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
