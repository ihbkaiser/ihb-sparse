"""Optional CUTLASS kernels for the ShadowKV decode path."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path


def _resolve_cutlass_root(cutlass_root: str | os.PathLike[str] | None) -> Path:
    value = cutlass_root or os.environ.get("SPARSEVLLM_CUTLASS_ROOT")
    if not value:
        raise RuntimeError(
            "ShadowKV CUTLASS backend requires SPARSEVLLM_CUTLASS_ROOT or "
            "an explicit cutlass_root argument pointing to a CUTLASS source tree."
        )
    root = Path(value).expanduser().resolve()
    if not (root / "include" / "cutlass").is_dir():
        raise FileNotFoundError(
            "ShadowKV CUTLASS root is missing include/cutlass: " f"{root}"
        )
    return root


@lru_cache(maxsize=4)
def load_shadowkv_cutlass(cutlass_root: str | None = None):
    """Build and load the CUTLASS ShadowKV extension once per root/process."""

    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("ShadowKV CUTLASS kernels require CUDA.")
    root = _resolve_cutlass_root(cutlass_root)
    source = Path(__file__).with_name("cuda") / "shadowkv_cutlass.cu"
    if not source.is_file():
        raise FileNotFoundError(f"ShadowKV CUTLASS source is missing: {source}")
    digest = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:10]
    module = load(
        name=f"sparsevllm_shadowkv_cutlass_{digest}",
        sources=[str(source)],
        extra_include_paths=[
            str(root / "include"),
            str(root / "tools" / "util" / "include"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    configure = getattr(module, "configure_shadowkv_cutlass", None)
    if configure is None:
        raise RuntimeError(
            "Loaded ShadowKV CUTLASS extension does not expose its pre-capture "
            "kernel configuration hook."
        )
    configure()
    return module


__all__ = ["load_shadowkv_cutlass"]
