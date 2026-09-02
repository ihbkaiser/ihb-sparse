"""Opt-in end-to-end smoke test for the Llama-3.1 ShadowKV path.

The test is deliberately opt-in because it downloads/loads an 8B checkpoint and
requires a CUDA-capable vLLM installation.  Run it from the repository root with
``RUN_SHADOWKV_E2E=1`` and optionally ``SHADOWKV_LLAMA_MODEL=/path/to/checkpoint``.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_shadowkv_llama31_ruler_smoke():
    if os.environ.get("RUN_SHADOWKV_E2E") != "1":
        pytest.skip("set RUN_SHADOWKV_E2E=1 to run the Llama-3.1 GPU smoke test")

    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    if not torch.cuda.is_available():
        pytest.skip("ShadowKV end-to-end smoke test requires CUDA")

    project_root = Path(__file__).resolve().parents[1]
    model_path = Path(
        os.environ.get(
            "SHADOWKV_LLAMA_MODEL",
            str(project_root / "experiments/checkpoints/Llama-3.1-8B-Instruct"),
        )
    ).expanduser().resolve()
    if not model_path.exists():
        pytest.skip(f"Llama-3.1 checkpoint is not available at {model_path}")

    command = [
        sys.executable,
        "-m",
        "sparse_frontier.main",
        "mode=all",
        "model=llama_8b",
        f"model.path={model_path}",
        "attention=shadowkv",
        "task=ruler_niah",
        "samples=1",
        "max_input_tokens=4096",
        "max_output_tokens=16",
        "gpus=1",
        "tp=1",
        "overwrite=true",
    ]
    result = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=int(os.environ.get("SHADOWKV_E2E_TIMEOUT", "1800")),
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "Llama-3.1 ShadowKV RULER smoke test failed\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )
