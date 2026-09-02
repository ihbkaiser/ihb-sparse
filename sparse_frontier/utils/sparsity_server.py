"""Small IPC server for attention-related metrics.

This server aggregates per-sample sparsity/cost statistics produced by vLLM
worker processes and consumed by the driver process.

Assumptions (by design for this repo's evaluation pipeline):
- batch size is always 1
- one active generation per process at a time
"""

from __future__ import annotations

import multiprocessing as mproc
from multiprocessing.connection import Client, Listener
from dataclasses import dataclass
from typing import Optional, Tuple, TypedDict
import os


ENV_ENABLED = "SF_SPARSITY_SERVER_ENABLED"
ENV_HOST = "SF_SPARSITY_SERVER_HOST"
ENV_PORT = "SF_SPARSITY_SERVER_PORT"
ENV_AUTHKEY = "SF_SPARSITY_SERVER_AUTHKEY"

@dataclass(frozen=True)
class SparsityServerConfig:
    host: str
    port: int
    authkey: str


@dataclass(frozen=True)
class SparsityServerHandle:
    process: mproc.Process
    config: SparsityServerConfig


class SparsityStats(TypedDict):
    prefill_sparsity: Optional[float]
    decode_access_sum: float
    decode_dense_sum: float
    shadowkv_state_ready: Optional[bool]
    shadowkv_layers_built: Optional[int]
    shadowkv_decode_seen: Optional[bool]
    query_robust_state_ready: Optional[bool]
    query_robust_summaries_built: Optional[int]
    query_robust_summary_build_ms: Optional[float]
    query_robust_fallback_count: Optional[int]
    query_robust_decode_seen: Optional[bool]
    query_robust_objective: Optional[str]
    query_robust_solver_gap_max: Optional[float]
    query_robust_solver_active_max: Optional[int]
    query_robust_solver_nonconverged: Optional[int]
    query_robust_quantized_error_max: Optional[float]


def _server_loop(address: Tuple[str, int], authkey: bytes, ready_event: mproc.Event) -> None:
    """Run the metrics server loop in a dedicated process."""
    listener = Listener(address, authkey=authkey, backlog=32)
    ready_event.set()

    prefill_sparsity: Optional[float] = None
    decode_access_sum = 0.0
    decode_dense_sum = 0.0
    shadowkv_state_ready: Optional[bool] = None
    shadowkv_layers_built: Optional[int] = None
    shadowkv_decode_seen: Optional[bool] = None
    query_robust_state_ready: Optional[bool] = None
    query_robust_summaries_built: Optional[int] = None
    query_robust_summary_build_ms: Optional[float] = None
    query_robust_fallback_count: Optional[int] = None
    query_robust_decode_seen: Optional[bool] = None
    query_robust_objective: Optional[str] = None
    query_robust_solver_gap_max: Optional[float] = None
    query_robust_solver_active_max: Optional[int] = None
    query_robust_solver_nonconverged: Optional[int] = None
    query_robust_quantized_error_max: Optional[float] = None

    try:
        while True:
            conn = listener.accept()
            try:
                message = conn.recv()
            except EOFError:
                conn.close()
                continue

            command = message.get("cmd") if isinstance(message, dict) else None

            if command == "sparsity_set_prefill":
                try:
                    value = float(message.get("value", 0.0))
                except (TypeError, ValueError):
                    conn.send({"status": "error", "error": "invalid value"})
                else:
                    prefill_sparsity = value
                    conn.send({"status": "ok"})
            elif command == "sparsity_add_decode":
                try:
                    accessed = float(message.get("accessed_sum", 0.0))
                    dense = float(message.get("dense_sum", 0.0))
                except (TypeError, ValueError):
                    conn.send({"status": "error", "error": "invalid decode payload"})
                else:
                    decode_access_sum += accessed
                    decode_dense_sum += dense
                    conn.send({"status": "ok"})
            elif command == "shadowkv_state":
                shadowkv_state_ready = bool(message.get("ready", False))
                shadowkv_layers_built = int(message.get("layers_built", 0))
                conn.send({"status": "ok"})
            elif command == "shadowkv_decode":
                shadowkv_decode_seen = True
                conn.send({"status": "ok"})
            elif command == "query_robust_state":
                query_robust_state_ready = bool(message.get("ready", False))
                query_robust_summaries_built = int(
                    message.get("summaries_built", 0)
                )
                query_robust_summary_build_ms = float(
                    message.get("summary_build_ms", 0.0)
                )
                query_robust_fallback_count = int(message.get("fallback_count", 0))
                query_robust_objective = str(message.get("objective", "minimax"))
                query_robust_solver_gap_max = float(message.get("solver_gap_max", 0.0))
                query_robust_solver_active_max = int(message.get("solver_active_max", 0))
                query_robust_solver_nonconverged = int(message.get("solver_nonconverged", 0))
                query_robust_quantized_error_max = float(message.get("quantized_error_max", 0.0))
                conn.send({"status": "ok"})
            elif command == "query_robust_decode":
                query_robust_decode_seen = True
                conn.send({"status": "ok"})
            elif command == "sparsity_fetch_and_reset":
                response = {
                    "status": "ok",
                    "prefill_sparsity": prefill_sparsity,
                    "decode_access_sum": decode_access_sum,
                    "decode_dense_sum": decode_dense_sum,
                    "shadowkv_state_ready": shadowkv_state_ready,
                    "shadowkv_layers_built": shadowkv_layers_built,
                    "shadowkv_decode_seen": shadowkv_decode_seen,
                    "query_robust_state_ready": query_robust_state_ready,
                    "query_robust_summaries_built": query_robust_summaries_built,
                    "query_robust_summary_build_ms": query_robust_summary_build_ms,
                    "query_robust_fallback_count": query_robust_fallback_count,
                    "query_robust_decode_seen": query_robust_decode_seen,
                    "query_robust_objective": query_robust_objective,
                    "query_robust_solver_gap_max": query_robust_solver_gap_max,
                    "query_robust_solver_active_max": query_robust_solver_active_max,
                    "query_robust_solver_nonconverged": query_robust_solver_nonconverged,
                    "query_robust_quantized_error_max": query_robust_quantized_error_max,
                }

                prefill_sparsity = None
                decode_access_sum = 0.0
                decode_dense_sum = 0.0
                shadowkv_state_ready = None
                shadowkv_layers_built = None
                shadowkv_decode_seen = None
                query_robust_state_ready = None
                query_robust_summaries_built = None
                query_robust_summary_build_ms = None
                query_robust_fallback_count = None
                query_robust_decode_seen = None
                query_robust_objective = None
                query_robust_solver_gap_max = None
                query_robust_solver_active_max = None
                query_robust_solver_nonconverged = None
                query_robust_quantized_error_max = None
                conn.send(response)
            else:
                conn.send({"status": "error", "error": "unknown command"})

            conn.close()
    finally:
        listener.close()


def start_sparsity_server(host: str, port: int, authkey: str) -> SparsityServerHandle:
    """Start the sparsity aggregation server in a background process."""
    ready_event = mproc.Event()
    process = mproc.Process(
        target=_server_loop,
        args=((host, port), authkey.encode("utf-8"), ready_event),
        daemon=True,
    )
    process.start()
    ready_event.wait()
    config = SparsityServerConfig(host=host, port=port, authkey=authkey)
    return SparsityServerHandle(process=process, config=config)


def configure_sparsity_env(config: SparsityServerConfig) -> None:
    """Expose server connection details to subprocesses via environment variables."""
    os.environ[ENV_ENABLED] = "1"
    os.environ[ENV_HOST] = config.host
    os.environ[ENV_PORT] = str(config.port)
    os.environ[ENV_AUTHKEY] = config.authkey


def set_prefill_sparsity(value: float) -> bool:
    """Set per-sample prefill sparsity (reported once per sample)."""
    resp = _send_command("sparsity_set_prefill", {"value": float(value)})
    return bool(resp and resp.get("status") == "ok")


def add_decode_access(accessed_sum: float, dense_sum: float) -> bool:
    """Add decode-time accessed-vs-dense token counts (reported every N steps)."""
    payload = {"accessed_sum": float(accessed_sum), "dense_sum": float(dense_sum)}
    resp = _send_command("sparsity_add_decode", payload)
    return bool(resp and resp.get("status") == "ok")


def set_shadowkv_state(ready: bool, layers_built: int) -> bool:
    """Record that all per-layer ShadowKV state was built for this request."""
    resp = _send_command(
        "shadowkv_state",
        {"ready": bool(ready), "layers_built": int(layers_built)},
    )
    return bool(resp and resp.get("status") == "ok")


def mark_shadowkv_decode() -> bool:
    """Record that the sparse ShadowKV decode path executed at least once."""
    resp = _send_command("shadowkv_decode")
    return bool(resp and resp.get("status") == "ok")


def set_query_robust_state(
    ready: bool,
    summaries_built: int,
    summary_build_ms: float,
    fallback_count: int,
    objective: str = "minimax",
    solver_gap_max: float = 0.0,
    solver_active_max: int = 0,
    solver_nonconverged: int = 0,
    quantized_error_max: float = 0.0,
) -> bool:
    """Record bounded Query-Robust request state after dense prefill."""
    resp = _send_command(
        "query_robust_state",
        {
            "ready": bool(ready),
            "summaries_built": int(summaries_built),
            "summary_build_ms": float(summary_build_ms),
            "fallback_count": int(fallback_count),
            "objective": str(objective),
            "solver_gap_max": float(solver_gap_max),
            "solver_active_max": int(solver_active_max),
            "solver_nonconverged": int(solver_nonconverged),
            "quantized_error_max": float(quantized_error_max),
        },
    )
    return bool(resp and resp.get("status") == "ok")


def mark_query_robust_decode() -> bool:
    """Record that Query-Robust exact-selected decode executed."""
    resp = _send_command("query_robust_decode")
    return bool(resp and resp.get("status") == "ok")


def fetch_sparsity_and_reset() -> Optional[SparsityStats]:
    """Fetch per-sample sparsity stats and reset accumulators."""
    response = _send_command("sparsity_fetch_and_reset", {})
    if not response or response.get("status") != "ok":
        return None
    return {
        "prefill_sparsity": response.get("prefill_sparsity"),
        "decode_access_sum": float(response.get("decode_access_sum", 0.0)),
        "decode_dense_sum": float(response.get("decode_dense_sum", 0.0)),
        "shadowkv_state_ready": response.get("shadowkv_state_ready"),
        "shadowkv_layers_built": response.get("shadowkv_layers_built"),
        "shadowkv_decode_seen": response.get("shadowkv_decode_seen"),
        "query_robust_state_ready": response.get("query_robust_state_ready"),
        "query_robust_summaries_built": response.get(
            "query_robust_summaries_built"
        ),
        "query_robust_summary_build_ms": response.get(
            "query_robust_summary_build_ms"
        ),
        "query_robust_fallback_count": response.get(
            "query_robust_fallback_count"
        ),
        "query_robust_decode_seen": response.get("query_robust_decode_seen"),
        "query_robust_objective": response.get("query_robust_objective"),
        "query_robust_solver_gap_max": response.get("query_robust_solver_gap_max"),
        "query_robust_solver_active_max": response.get("query_robust_solver_active_max"),
        "query_robust_solver_nonconverged": response.get("query_robust_solver_nonconverged"),
        "query_robust_quantized_error_max": response.get("query_robust_quantized_error_max"),
    }


def _connect() -> Optional[Client]:
    if os.environ.get(ENV_ENABLED) != "1":
        return None

    host = os.environ.get(ENV_HOST)
    port = os.environ.get(ENV_PORT)
    authkey = os.environ.get(ENV_AUTHKEY)
    if not host or not port or not authkey:
        return None

    try:
        connection = Client((host, int(port)), authkey=authkey.encode("utf-8"))
    except OSError:
        return None

    return connection


def _send_command(command: str, payload: Optional[dict] = None) -> Optional[dict]:
    conn = _connect()
    if conn is None:
        return None

    try:
        message = {"cmd": command}
        if payload:
            message.update(payload)
        conn.send(message)
        return conn.recv()
    except (OSError, EOFError):
        return None
    finally:
        conn.close()
