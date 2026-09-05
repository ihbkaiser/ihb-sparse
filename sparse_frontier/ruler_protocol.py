"""Immutable 128K RULER contracts for controlled sparse-attention runs.

The Hugging Face KVPress export in this repository only contains short
contexts.  A 128K experiment therefore consumes a locally generated RULER
JSONL file, but never an ad-hoc collection of prompts: this module checks the
source revision and the digest of the exact byte stream before any method sees
the rows.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sparse_frontier.ruler_kvpress import RULER_TASKS


TABLE1_PROTOCOL = "ruler_128k_table1_recipe"
TABLE1_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multiquery",
    "niah_multivalue",
    "qa_1",
    "qa_2",
    "vt",
    "fwe",
)
_HEX = re.compile(r"^[0-9a-f]+$")


@dataclass(frozen=True)
class Table1Protocol:
    """Validated, immutable input contract shared by every benchmark method."""

    manifest_path: Path
    rows_path: Path
    rows_sha256: str
    source: dict[str, Any]
    context_length: int
    seed: int
    task_set: tuple[str, ...]
    samples_per_task: int
    model: dict[str, str]
    tokenizer: dict[str, str]
    runtime: dict[str, Any]
    hardware_policy: dict[str, Any]
    rows: list[dict[str, Any]]


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"protocol manifest field {name!r} must be an object")
    return value


def _require_sha(value: Any, name: str, length: int = 64) -> str:
    text = str(value)
    if len(text) != length or _HEX.fullmatch(text) is None:
        raise ValueError(f"protocol manifest field {name!r} must be a {length}-hex revision/digest")
    return text


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read protocol manifest {path}: {exc}") from exc
    return _require_mapping(value, "root")


def _read_rows(path: Path, expected_sha256: str) -> list[Mapping[str, Any]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read protocol rows {path}: {exc}") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"protocol rows SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"protocol rows contain a blank line at {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"protocol rows contain invalid JSON at line {line_number}") from exc
        rows.append(_require_mapping(row, f"row {line_number}"))
    if not rows:
        raise ValueError("protocol rows are empty")
    return rows


def _answers(value: Any, *, row_index: int) -> list[str]:
    if isinstance(value, str):
        result = [value]
    elif isinstance(value, list):
        result = [str(item) for item in value]
    else:
        raise ValueError(f"protocol row {row_index} answers must be a string or list of strings")
    if not result:
        raise ValueError(f"protocol row {row_index} must contain at least one answer")
    return result


def _normalize_row(
    source: Mapping[str, Any],
    *,
    index: int,
    context_length: int,
    output_caps: Mapping[str, int],
    task_index: int,
) -> dict[str, Any]:
    task = str(source.get("task", ""))
    if task not in TABLE1_TASKS:
        raise ValueError(f"protocol row {index} has task outside the locked Table-1 task set: {task!r}")
    answers = _answers(source.get("answer", source.get("outputs")), row_index=index)
    expected_cap = int(output_caps[task])
    declared_cap = int(source.get("max_new_tokens", expected_cap))
    if declared_cap != expected_cap:
        raise ValueError(
            f"protocol row {index} max_new_tokens={declared_cap} does not match locked "
            f"task output cap {expected_cap} for {task}"
        )
    row = {
        "index": int(source.get("index", index)),
        "task": task,
        "task_index": task_index,
        "answer": answers,
        "gold_answer": answers,
        "max_new_tokens": expected_cap,
        "tokens_to_generate": expected_cap,
        "context_length": context_length,
    }
    # Upstream RULER generators serialize a rendered ``input``.  Retokenizing
    # it is intentional; applying a chat template to it would be a benchmark
    # corruption, so the prompt builder checks this explicit flag.
    if "input" in source:
        row.update(
            {
                "prompt": str(source["input"]),
                "prompt_is_preformatted": True,
                "context": "",
                "question": "",
                "answer_prefix": "",
            }
        )
    else:
        required = ("context", "question", "answer_prefix")
        missing = [field for field in required if field not in source]
        if missing:
            raise ValueError(f"protocol row {index} is missing fields: {', '.join(missing)}")
        row.update(
            {
                "context": str(source["context"]),
                "question": str(source["question"]),
                "answer_prefix": str(source["answer_prefix"]),
            }
        )
    return row


def _validate_runtime(runtime: Mapping[str, Any], hardware_policy: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    required_runtime = {
        "dtype",
        "greedy",
        "batch_size",
        "prefill",
        "max_input_tokens",
        "max_output_tokens",
        "task_output_caps",
    }
    missing_runtime = sorted(required_runtime - set(runtime))
    if missing_runtime:
        raise ValueError(f"protocol runtime is missing fields: {', '.join(missing_runtime)}")
    normalized_runtime = dict(runtime)
    if normalized_runtime["dtype"] != "bfloat16":
        raise ValueError("Table-1 protocol requires dtype=bfloat16")
    if normalized_runtime["greedy"] is not True:
        raise ValueError("Table-1 protocol requires greedy decoding")
    if int(normalized_runtime["batch_size"]) != 1:
        raise ValueError("Table-1 protocol requires batch_size=1")
    if normalized_runtime["prefill"] != "dense_exact":
        raise ValueError("Table-1 protocol requires dense_exact prefill")
    if int(normalized_runtime["max_input_tokens"]) != 131072:
        raise ValueError("Table-1 protocol requires max_input_tokens=131072")
    if int(normalized_runtime["max_output_tokens"]) < 1:
        raise ValueError("protocol max_output_tokens must be positive")
    caps = _require_mapping(normalized_runtime["task_output_caps"], "runtime.task_output_caps")
    if set(caps) != set(TABLE1_TASKS):
        raise ValueError("protocol task_output_caps must cover exactly the locked Table-1 task set")
    normalized_caps = {task: int(caps[task]) for task in TABLE1_TASKS}
    if any(value < 1 or value > int(normalized_runtime["max_output_tokens"]) for value in normalized_caps.values()):
        raise ValueError("protocol task output caps must be positive and no greater than max_output_tokens")
    normalized_runtime["task_output_caps"] = normalized_caps

    required_hardware = {
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "cpu_offload_gb",
        "max_num_batched_tokens",
    }
    missing_hardware = sorted(required_hardware - set(hardware_policy))
    if missing_hardware:
        raise ValueError(f"protocol hardware_policy is missing fields: {', '.join(missing_hardware)}")
    normalized_hardware = dict(hardware_policy)
    if int(normalized_hardware["tensor_parallel_size"]) < 1:
        raise ValueError("protocol tensor_parallel_size must be positive")
    utilization = float(normalized_hardware["gpu_memory_utilization"])
    if not 0.0 < utilization <= 1.0:
        raise ValueError("protocol gpu_memory_utilization must be in (0, 1]")
    if float(normalized_hardware["cpu_offload_gb"]) < 0:
        raise ValueError("protocol cpu_offload_gb must be non-negative")
    if int(normalized_hardware["max_num_batched_tokens"]) < (
        int(normalized_runtime["max_input_tokens"]) + int(normalized_runtime["max_output_tokens"])
    ):
        raise ValueError("protocol max_num_batched_tokens must keep the full prefill unchunked")
    return normalized_runtime, normalized_hardware


def load_table1_protocol(manifest_path: str | Path) -> Table1Protocol:
    """Load a digest-checked 128K Table-1-compatible RULER contract."""
    path = Path(manifest_path).resolve()
    manifest = _read_json(path)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported protocol manifest schema_version")
    if manifest.get("protocol") != TABLE1_PROTOCOL:
        raise ValueError(f"protocol must be {TABLE1_PROTOCOL!r}")
    source = _require_mapping(manifest.get("source"), "source")
    for field in ("name", "repository", "revision"):
        if field not in source or not str(source[field]):
            raise ValueError(f"protocol source is missing {field!r}")
    _require_sha(source["revision"], "source.revision", length=40)
    rows_sha256 = _require_sha(manifest.get("rows_sha256"), "rows_sha256")
    rows_path_value = manifest.get("rows_path")
    if not isinstance(rows_path_value, str) or not rows_path_value:
        raise ValueError("protocol rows_path must be a nonempty relative path")
    rows_path = (path.parent / rows_path_value).resolve()
    if path.parent not in rows_path.parents:
        raise ValueError("protocol rows_path must stay below the manifest directory")
    if int(manifest.get("context_length", 0)) != 131072:
        raise ValueError("Table-1 protocol requires context_length=131072")
    task_set = tuple(str(task) for task in manifest.get("task_set", ()))
    if task_set != TABLE1_TASKS:
        raise ValueError("protocol task_set must equal the ordered Table-1 task set")
    samples_per_task = int(manifest.get("samples_per_task", 0))
    if samples_per_task < 1:
        raise ValueError("protocol samples_per_task must be positive")
    model = _require_mapping(manifest.get("model"), "model")
    tokenizer = _require_mapping(manifest.get("tokenizer"), "tokenizer")
    for name, value in (("model", model), ("tokenizer", tokenizer)):
        for field in ("id", "revision"):
            if field not in value or not str(value[field]):
                raise ValueError(f"protocol {name} is missing {field!r}")
        _require_sha(value["revision"], f"{name}.revision", length=40)
    if "template_sha256" not in tokenizer:
        raise ValueError("protocol tokenizer is missing 'template_sha256'")
    _require_sha(tokenizer["template_sha256"], "tokenizer.template_sha256")
    runtime, hardware_policy = _validate_runtime(
        _require_mapping(manifest.get("runtime"), "runtime"),
        _require_mapping(manifest.get("hardware_policy"), "hardware_policy"),
    )

    raw_rows = _read_rows(rows_path, rows_sha256)
    task_indexes: Counter[str] = Counter()
    rows = []
    for index, raw in enumerate(raw_rows):
        task = str(raw.get("task", ""))
        row = _normalize_row(
            raw,
            index=index,
            context_length=131072,
            output_caps=runtime["task_output_caps"],
            task_index=task_indexes[task],
        )
        rows.append(row)
        task_indexes[task] += 1
    observed = {task: task_indexes[task] for task in TABLE1_TASKS}
    expected = {task: samples_per_task for task in TABLE1_TASKS}
    if observed != expected:
        raise ValueError(f"protocol task matrix differs from samples_per_task: expected {expected}, got {observed}")

    return Table1Protocol(
        manifest_path=path,
        rows_path=rows_path,
        rows_sha256=rows_sha256,
        source=dict(source),
        context_length=131072,
        seed=int(manifest.get("seed")),
        task_set=task_set,
        samples_per_task=samples_per_task,
        model={"id": str(model["id"]), "revision": str(model["revision"])},
        tokenizer={
            "id": str(tokenizer["id"]),
            "revision": str(tokenizer["revision"]),
            "template_sha256": str(tokenizer["template_sha256"]),
        },
        runtime=runtime,
        hardware_policy=hardware_policy,
        rows=rows,
    )


def validate_pile_only_query_pool(query_pool_path: str | Path) -> Path:
    """Reject query support captured from a RULER request or another test set.

    ``vllm_empirical_query_pool`` can be finalized from arbitrary dense
    captures, including evaluation prompts.  The locked protocol accepts only
    the direct Pile capture artifact, whose manifest records its external
    train split and revision.
    """
    root = Path(query_pool_path).resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Pile query-pool manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != 2 or manifest.get("artifact_type") != "pile_empirical_query_pool":
        raise ValueError(
            "Table-1 protocol requires a Pile-only query pool; request-derived or vLLM capture pools are forbidden"
        )
    dataset = _require_mapping(manifest.get("dataset"), "query-pool dataset")
    if not str(dataset.get("dataset", "")) or str(dataset.get("split", "")) != "train":
        raise ValueError("Pile-only query pool must record a nonempty dataset and split=train")
    return root


def validate_table1_model_contract(contract: Table1Protocol, model_cfg: Mapping[str, Any]) -> None:
    """Fail before inference when a snapshot differs from the pinned backbone."""
    actual_id = str(model_cfg.get("model_id", ""))
    actual_revision = str(model_cfg.get("model_revision", ""))
    if actual_id != contract.model["id"]:
        raise ValueError(
            f"protocol model identifier mismatch: expected {contract.model['id']!r}, got {actual_id!r}"
        )
    if actual_revision != contract.model["revision"]:
        raise ValueError(
            f"protocol model revision mismatch: expected {contract.model['revision']!r}, got {actual_revision!r}"
        )


def validate_table1_tokenizer_contract(contract: Table1Protocol, tokenizer: Any) -> None:
    """Check the rendered-template identity before rows are tokenized."""
    template = getattr(tokenizer, "chat_template", None)
    actual_sha256 = hashlib.sha256(str(template or "").encode("utf-8")).hexdigest()
    if actual_sha256 != contract.tokenizer["template_sha256"]:
        raise ValueError(
            "protocol tokenizer template SHA256 mismatch: "
            f"expected {contract.tokenizer['template_sha256']}, got {actual_sha256}"
        )


__all__ = [
    "TABLE1_PROTOCOL",
    "TABLE1_TASKS",
    "Table1Protocol",
    "load_table1_protocol",
    "validate_pile_only_query_pool",
    "validate_table1_model_contract",
    "validate_table1_tokenizer_contract",
]
