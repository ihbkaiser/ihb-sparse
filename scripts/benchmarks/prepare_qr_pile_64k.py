"""Materialize the deterministic 64K Pile calibration contexts as JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _pack_contexts(
    tokenizer: Any,
    dataset: Any,
    *,
    dataset_name: str,
    dataset_revision: str,
    num_sequences: int,
    sequence_length: int,
    eos_token_id: int,
) -> tuple[list[list[int]], list[list[dict[str, Any]]]]:
    contexts: list[list[int]] = []
    source_manifest: list[list[dict[str, Any]]] = []
    current: list[int] = []
    current_sources: list[dict[str, Any]] = []

    for source_index, row in enumerate(dataset):
        text = str(row.get("text", ""))
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        token_ids = [*token_ids, int(eos_token_id)]
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        offset = 0
        while offset < len(token_ids):
            remaining = sequence_length - len(current)
            consumed = min(remaining, len(token_ids) - offset)
            current.extend(token_ids[offset : offset + consumed])
            current_sources.append(
                {
                    "dataset": dataset_name,
                    "dataset_revision": dataset_revision,
                    "stream_index": source_index,
                    "text_sha256": text_hash,
                    "token_start": offset,
                    "token_end": offset + consumed,
                }
            )
            offset += consumed
            if len(current) == sequence_length:
                contexts.append(current)
                source_manifest.append(current_sources)
                if len(contexts) == num_sequences:
                    return contexts, source_manifest
                current = []
                current_sources = []

    raise RuntimeError(
        f"Only packed {len(contexts)} of {num_sequences} required contexts."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", default="monology/pile-uncopyrighted")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--num-sequences", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=65_536)
    args = parser.parse_args()

    if args.num_sequences <= 0 or args.sequence_length <= 0:
        raise ValueError("num-sequences and sequence-length must be positive.")

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id.")

    revision = args.revision or HfApi().dataset_info(args.dataset).sha
    dataset = load_dataset(
        args.dataset,
        revision=revision,
        split=args.split,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
    contexts, source_manifest = _pack_contexts(
        tokenizer,
        dataset,
        dataset_name=args.dataset,
        dataset_revision=revision,
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        eos_token_id=eos_token_id,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, input_ids in enumerate(contexts):
            record = {
                "sequence_index": index,
                "input_ids": input_ids,
                "context_length": len(input_ids),
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    manifest = {
        "dataset": args.dataset,
        "dataset_revision": revision,
        "split": args.split,
        "seed": args.seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "num_sequences": len(contexts),
        "sequence_length": args.sequence_length,
        "tokenizer_path": str(model_path),
        "eos_token_id": int(eos_token_id),
        "contexts": source_manifest,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "success",
        "output": str(args.output),
        "manifest": str(args.manifest),
        "dataset_revision": revision,
        "num_sequences": len(contexts),
        "sequence_length": args.sequence_length,
    }))


if __name__ == "__main__":
    main()
