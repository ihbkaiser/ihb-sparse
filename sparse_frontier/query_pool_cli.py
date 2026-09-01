"""Command-line lifecycle tools for Query-Robust empirical query pools."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from sparse_frontier.modelling.attention.query_pool import (
    QueryPoolManifest,
    finalize_capture,
)


def _read_manifest(pool_path: str | Path) -> QueryPoolManifest:
    path = Path(pool_path) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"query-pool manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read query-pool manifest {path}: {exc}") from exc
    return QueryPoolManifest.from_dict(payload)


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def inspect_pool(pool_path: str | Path) -> dict[str, Any]:
    root = Path(pool_path)
    manifest = _read_manifest(root)
    online_bytes = 0
    offline_bytes = 0
    for rank in range(manifest.tp_size):
        for layer_idx in range(manifest.num_layers):
            online_path = root / f"layer_{layer_idx:03d}_rank_{rank:03d}.pt"
            if not online_path.is_file():
                raise FileNotFoundError(f"query-pool online shard is missing: {online_path}")
            try:
                online = torch.load(online_path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise ValueError(f"cannot load query-pool online shard {online_path}: {exc}") from exc
            online_bytes += _tensor_bytes(online)
            full_path = root / "full" / f"layer_{layer_idx:03d}_rank_{rank:03d}.pt"
            if full_path.exists():
                try:
                    full = torch.load(full_path, map_location="cpu", weights_only=True)
                except Exception as exc:
                    raise ValueError(f"cannot load query-pool full shard {full_path}: {exc}") from exc
                offline_bytes += _tensor_bytes(full)
    return {
        **manifest.to_dict(),
        "online_query_bytes": online_bytes,
        "offline_full_pool_bytes": offline_bytes,
        "status": "ok",
    }


def validate_model(pool_path: str | Path, model_path: str | Path) -> dict[str, Any]:
    manifest = _read_manifest(pool_path)
    model_root = Path(model_path)
    config_path = model_root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config is missing: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read model config {config_path}: {exc}") from exc

    expected = {
        "num_layers": int(config["num_hidden_layers"]),
        "num_q_heads": int(config["num_attention_heads"]),
        "num_kv_heads": int(config["num_key_value_heads"]),
        "head_dim": int(config["hidden_size"]) // int(config["num_attention_heads"]),
    }
    for field, value in expected.items():
        if getattr(manifest, field) != value:
            raise ValueError(
                f"query-pool {field.replace('_', ' ')} mismatch: "
                f"artifact={getattr(manifest, field)}, model={value}"
            )
    snapshot_revision = model_root.resolve().name
    if len(snapshot_revision) == 40 and snapshot_revision != manifest.model_revision:
        raise ValueError(
            "query-pool model revision mismatch: "
            f"artifact={manifest.model_revision}, snapshot={snapshot_revision}"
        )
    rope_scaling = config.get("rope_scaling") or {}
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
    if manifest.rope_type != rope_type:
        raise ValueError(
            f"query-pool rope type mismatch: artifact={manifest.rope_type}, model={rope_type}"
        )
    available_rope = {**rope_scaling, "rope_theta": config.get("rope_theta")}
    for key, value in manifest.rope_parameters.items():
        if available_rope.get(key) != value:
            raise ValueError(
                f"query-pool RoPE parameter {key} mismatch: "
                f"artifact={value!r}, model={available_rope.get(key)!r}"
            )
    if config.get("attention_bias", False):
        raise ValueError("query-pool validation rejects additive attention bias")
    expected_scale = expected["head_dim"] ** -0.5
    if not math.isclose(manifest.attention_scale, expected_scale, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(
            f"query-pool attention scale mismatch: artifact={manifest.attention_scale}, "
            f"model={expected_scale}"
        )
    result = inspect_pool(pool_path)
    result.update({"model_path": str(model_root), "status": "valid"})
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect a frozen pool")
    inspect_parser.add_argument("--pool", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate against a model")
    validate_parser.add_argument("--pool", required=True)
    validate_parser.add_argument("--model_path", required=True)

    finalize_parser = subparsers.add_parser(
        "capture-finalize", help="finalize raw capture shards"
    )
    finalize_parser.add_argument("--input", required=True)
    finalize_parser.add_argument("--output", required=True)
    finalize_parser.add_argument("--pool_size", type=int, required=True)
    finalize_parser.add_argument("--coreset_sizes", type=int, nargs="*", default=[])
    finalize_parser.add_argument("--seed", type=int, default=43)
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_pool(args.pool)
        elif args.command == "validate":
            result = validate_model(args.pool, args.model_path)
        else:
            output = finalize_capture(
                input_dir=args.input,
                output_dir=args.output,
                pool_size=args.pool_size,
                coreset_sizes=args.coreset_sizes,
                seed=args.seed,
            )
            result = {"status": "ok", "pool": str(output)}
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
