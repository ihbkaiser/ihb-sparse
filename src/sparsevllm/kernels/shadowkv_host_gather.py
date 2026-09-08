"""Lazy loader for the CUDA-graph-safe ShadowKV host gather kernel."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_shadowkv_host_gather():
    """Build and load the small extension exactly once per Python process."""
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("ShadowKV host gather requires CUDA.")
    source = Path(__file__).with_name("cuda") / "shadowkv_host_gather.cu"
    if not source.is_file():
        raise FileNotFoundError(f"ShadowKV CUDA source is missing: {source}")
    return load(
        name="sparsevllm_shadowkv_host_gather_v5",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


__all__ = ["load_shadowkv_host_gather"]
