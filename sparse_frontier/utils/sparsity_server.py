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


def _server_loop(address: Tuple[str, int], authkey: bytes, ready_event: mproc.Event) -> None:
    """Run the metrics server loop in a dedicated process."""
    listener = Listener(address, authkey=authkey, backlog=32)
    ready_event.set()

    prefill_sparsity: Optional[float] = None
    decode_access_sum = 0.0
    decode_dense_sum = 0.0

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
            elif command == "sparsity_fetch_and_reset":
                response = {
                    "status": "ok",
                    "prefill_sparsity": prefill_sparsity,
                    "decode_access_sum": decode_access_sum,
                    "decode_dense_sum": decode_dense_sum,
                }

                prefill_sparsity = None
                decode_access_sum = 0.0
                decode_dense_sum = 0.0
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


def fetch_sparsity_and_reset() -> Optional[SparsityStats]:
    """Fetch per-sample sparsity stats and reset accumulators."""
    response = _send_command("sparsity_fetch_and_reset", {})
    if not response or response.get("status") != "ok":
        return None
    return {
        "prefill_sparsity": response.get("prefill_sparsity"),
        "decode_access_sum": float(response.get("decode_access_sum", 0.0)),
        "decode_dense_sum": float(response.get("decode_dense_sum", 0.0)),
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

