"""Build a final Query-Robust M32 asset from tokenized 128K contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


CALIBRATION_SEED = 20260911
NUM_SEQUENCES = 20
SEQUENCE_LENGTH = 131_072
SAMPLES_PER_QUERY_HEAD = 3_000
NUM_LAYERS = 32
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_VERTICES = 32
POSITION_WINDOWS = {
    f"{16 * index}k": (16_384 * index - 2_048, 16_384 * index)
    for index in range(1, 9)
}


def _read_contexts(path: Path) -> list[torch.Tensor]:
    contexts: list[torch.Tensor] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            input_ids = record.get("input_ids")
            if not isinstance(input_ids, list):
                raise ValueError(f"{path}:{line_number}: input_ids must be a list.")
            if len(input_ids) != SEQUENCE_LENGTH:
                raise ValueError(
                    f"{path}:{line_number}: expected {SEQUENCE_LENGTH} tokens, "
                    f"got {len(input_ids)}."
                )
            contexts.append(torch.tensor(input_ids, dtype=torch.long))
    if len(contexts) != NUM_SEQUENCES:
        raise ValueError(
            f"Expected exactly {NUM_SEQUENCES} contexts, got {len(contexts)}."
        )
    return contexts


def _validate_calibration_manifest(
    calibration_jsonl: Path, manifest_path: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("num_sequences") != NUM_SEQUENCES:
        raise ValueError("Calibration manifest has an unexpected sequence count.")
    if manifest.get("sequence_length") != SEQUENCE_LENGTH:
        raise ValueError("Calibration manifest is not a 128K calibration input.")
    if manifest.get("samples_per_query_head") != SAMPLES_PER_QUERY_HEAD:
        raise ValueError("Calibration manifest has an unexpected sample budget.")
    expected_windows = {
        label: list(window) for label, window in POSITION_WINDOWS.items()
    }
    if manifest.get("position_windows") != expected_windows:
        raise ValueError("Calibration position windows do not match this builder.")
    actual_sha256 = hashlib.sha256(calibration_jsonl.read_bytes()).hexdigest()
    if actual_sha256 != manifest.get("jsonl_sha256"):
        raise ValueError(
            "Calibration JSONL SHA-256 does not match its manifest: "
            f"expected={manifest.get('jsonl_sha256')!r} actual={actual_sha256!r}."
        )
    return manifest


def _capture_queries(
    model_path: Path,
    contexts: list[torch.Tensor],
    *,
    seed: int,
    model_id: str,
    model_fingerprint: str,
    output_path: Path,
) -> tuple[Path, dict[str, Any]]:
    from transformers import AutoModelForCausalLM
    from transformers.models.llama import modeling_llama

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from sparsevllm.models.rope import resolve_rope_theta
    from tools.query_robust.final_calibration import allocate_stratified_queries

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for 128K calibration.")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    decoder = getattr(model, "model", None)
    if decoder is None or not hasattr(decoder, "layers"):
        raise RuntimeError("Expected a Llama decoder with model.layers.")
    if len(decoder.layers) != NUM_LAYERS:
        raise RuntimeError(f"Expected {NUM_LAYERS} layers, got {len(decoder.layers)}.")

    allocation = allocate_stratified_queries(
        total_samples=SAMPLES_PER_QUERY_HEAD,
        num_sequences=NUM_SEQUENCES,
        buckets=tuple(POSITION_WINDOWS),
    )
    captured: list[list[torch.Tensor]] = [[] for _ in range(NUM_LAYERS)]
    active_layer = [-1]
    active_positions: list[torch.Tensor | None] = [None]
    original_apply = modeling_llama.apply_rotary_pos_emb
    original_forwards = [layer.self_attn.forward for layer in decoder.layers]

    def capture_rotary(query_states, key_states, cos, sin, *args, **kwargs):
        rotated_q, rotated_k = original_apply(
            query_states, key_states, cos, sin, *args, **kwargs
        )
        positions = active_positions[0]
        if active_layer[0] >= 0 and positions is not None:
            captured[active_layer[0]].append(
                rotated_q[0, :, positions, :].detach().to(torch.bfloat16).cpu()
            )
        return rotated_q, rotated_k

    modeling_llama.apply_rotary_pos_emb = capture_rotary
    for layer_index, layer in enumerate(decoder.layers):
        original_forward = original_forwards[layer_index]

        def wrapped_forward(
            *args, _idx=layer_index, _forward=original_forward, **kwargs
        ):
            active_layer[0] = _idx
            return _forward(*args, **kwargs)

        layer.self_attn.forward = wrapped_forward

    labels: list[str] = []
    context_lengths: list[int] = []
    decode_steps: list[int] = []
    expected_per_bucket = SAMPLES_PER_QUERY_HEAD // len(POSITION_WINDOWS)
    try:
        with torch.inference_mode():
            for sequence_index, context in enumerate(contexts):
                selected: list[torch.Tensor] = []
                for bucket_index, (label, (start, end)) in enumerate(
                    POSITION_WINDOWS.items()
                ):
                    count = allocation[(sequence_index, label)]
                    generator = torch.Generator(device="cpu").manual_seed(
                        seed + sequence_index * 97 + bucket_index
                    )
                    positions = torch.randperm(end - start, generator=generator)[:count]
                    positions = positions.add(start)
                    selected.append(positions)
                    labels.extend([label] * count)
                    context_lengths.extend([end] * count)
                    decode_steps.extend(positions.tolist())
                active_positions[0] = torch.cat(selected).to(device)
                input_ids = context.to(device=device).unsqueeze(0)
                decoder(input_ids, use_cache=False, return_dict=True)
                active_positions[0] = None
                print(
                    f"QR_128K_SEQUENCE={sequence_index + 1}/{NUM_SEQUENCES}",
                    flush=True,
                )
    finally:
        active_positions[0] = None
        modeling_llama.apply_rotary_pos_emb = original_apply
        for layer, original_forward in zip(
            decoder.layers, original_forwards, strict=True
        ):
            layer.self_attn.forward = original_forward

    if any(len(rows) != NUM_SEQUENCES for rows in captured):
        raise RuntimeError("Every layer must capture every context.")
    if any(
        labels.count(label) != expected_per_bucket for label in POSITION_WINDOWS
    ):
        raise RuntimeError("The 128K calibration buckets are not balanced.")

    per_layer = [torch.cat(rows, dim=1).contiguous() for rows in captured]
    queries = torch.stack(per_layer, dim=1).permute(2, 1, 0, 3).contiguous()
    expected_shape = (SAMPLES_PER_QUERY_HEAD, NUM_LAYERS, NUM_QUERY_HEADS, HEAD_DIM)
    if tuple(queries.shape) != expected_shape:
        raise RuntimeError(f"Expected queries {expected_shape}, got {tuple(queries.shape)}.")

    rope_config = getattr(model.config, "rope_parameters", None)
    if rope_config is None:
        rope_config = getattr(model.config, "rope_scaling", None)
    if rope_config is not None:
        rope_config = dict(rope_config)
        rope_config["rope_theta"] = resolve_rope_theta(model.config)
    payload = {
        "queries": queries,
        "context_lengths": torch.tensor(context_lengths, dtype=torch.int64),
        "context_buckets": labels,
        "decode_steps": torch.tensor(decode_steps, dtype=torch.int64),
        "meta": {
            "model_id": model_id,
            "model_fingerprint": model_fingerprint,
            "capture": "final Pile-only post-RoPE Query-Robust 128K calibration",
            "num_sequences": NUM_SEQUENCES,
            "sequence_length": SEQUENCE_LENGTH,
            "num_svd_samples": SAMPLES_PER_QUERY_HEAD,
            "samples_per_query_head": SAMPLES_PER_QUERY_HEAD,
            "samples_per_bucket_per_query_head": expected_per_bucket,
            "position_windows": POSITION_WINDOWS,
            "tp_world_size": 1,
            "rope_config": rope_config,
            "packing": "tokenized Pile contexts with EOS to exact 131072 tokens",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    del model
    torch.cuda.empty_cache()
    return output_path, payload["meta"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEED)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.query_robust.collect_queries import collect
    from tools.query_robust.select_vertices import select_vertices
    from sparsevllm.engine.cache_manager.query_robust import load_query_robust_asset

    manifest = _validate_calibration_manifest(
        args.calibration_jsonl, args.calibration_manifest
    )
    contexts = _read_contexts(args.calibration_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture_path, capture_meta = _capture_queries(
        args.model_path,
        contexts,
        seed=args.seed,
        model_id=args.model_id,
        model_fingerprint=args.model_fingerprint,
        output_path=args.output_dir / "pile128k_q_capture.pt",
    )
    collected_path = args.output_dir / "pile128k_collected.pt"
    collect(
        [capture_path],
        output=collected_path,
        num_kv_heads=NUM_KV_HEADS,
        max_per_group=expected_max_per_group(),
        seed=args.seed,
        extra_meta={
            "model_id": args.model_id,
            "model_fingerprint": args.model_fingerprint,
            "tp_world_size": 1,
            "rope_config": capture_meta["rope_config"],
            "calibration_jsonl_sha256": manifest["jsonl_sha256"],
        },
    )
    asset_path = args.output_dir / "qr_vertices_m32_128k.pt"
    select_vertices(
        collected_path,
        asset_path,
        num_vertices=NUM_VERTICES,
        model_id=args.model_id,
        model_fingerprint=args.model_fingerprint,
        tp_world_size=1,
        rope_config=capture_meta["rope_config"],
        device="cuda",
    )
    asset = load_query_robust_asset(
        asset_path,
        num_layers=NUM_LAYERS,
        global_num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        num_vertices=NUM_VERTICES,
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        expected_model_id=args.model_id,
        expected_model_fingerprint=args.model_fingerprint,
        expected_rope_config=capture_meta["rope_config"],
    )
    if not torch.equal(
        asset.num_valid_vertices,
        torch.full_like(asset.num_valid_vertices, NUM_VERTICES),
    ):
        raise RuntimeError("The final 128K asset does not have 32 valid vertices everywhere.")
    digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    output_manifest = {
        "status": "success",
        "asset": str(asset_path),
        "sha256": digest,
        "shape": list(asset.vertices.shape),
        "dtype": str(asset.vertices.dtype),
        "model_id": args.model_id,
        "model_fingerprint": args.model_fingerprint,
        "num_sequences": NUM_SEQUENCES,
        "sequence_length": SEQUENCE_LENGTH,
        "samples_per_query_head": SAMPLES_PER_QUERY_HEAD,
        "samples_per_bucket_per_query_head": SAMPLES_PER_QUERY_HEAD // len(POSITION_WINDOWS),
        "position_windows": POSITION_WINDOWS,
        "num_vertices": NUM_VERTICES,
        "calibration_jsonl_sha256": manifest["jsonl_sha256"],
    }
    (args.output_dir / "calibration_manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output_manifest, indent=2))


def expected_max_per_group() -> int:
    """Keep every captured sample in each (layer, KV head, Q head, bucket) group."""

    return SAMPLES_PER_QUERY_HEAD // len(POSITION_WINDOWS)


if __name__ == "__main__":
    main()
