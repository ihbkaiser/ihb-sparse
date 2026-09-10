"""Lazy loader for the CUDA-graph-safe ShadowKV host gather kernel."""

from __future__ import annotations

import importlib.util
import platform
from functools import lru_cache
from pathlib import Path


def _cuda_include_paths(cuda_home: str | None) -> list[str]:
    """Find CUDA development headers used by PyTorch CUDA extensions."""
    candidates: list[Path] = []
    if cuda_home:
        root = Path(cuda_home).expanduser()
        candidates.extend(
            [
                root / "include",
                root / "targets" / f"{platform.machine()}-linux" / "include",
            ]
        )

    spec = importlib.util.find_spec("nvidia.cusparse")
    if spec and spec.submodule_search_locations:
        candidates.extend(
            Path(location) / "include"
            for location in spec.submodule_search_locations
        )

    include_paths: list[str] = []
    for candidate in candidates:
        if (candidate / "cusparse.h").is_file():
            resolved = str(candidate.resolve())
            if resolved not in include_paths:
                include_paths.append(resolved)
    return include_paths


@lru_cache(maxsize=1)
def load_shadowkv_host_gather():
    """Build and load the small extension exactly once per Python process."""
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if not torch.cuda.is_available():
        raise RuntimeError("ShadowKV host gather requires CUDA.")
    source = Path(__file__).with_name("cuda") / "shadowkv_host_gather.cu"
    if not source.is_file():
        raise FileNotFoundError(f"ShadowKV CUDA source is missing: {source}")
    cuda_include_paths = _cuda_include_paths(CUDA_HOME)
    if not cuda_include_paths:
        raise RuntimeError(
            "ShadowKV host gather requires the CUDA cuSPARSE development header "
            "cusparse.h. Install a CUDA development toolkit or the matching "
            "nvidia-cusparse package, then retry."
        )
    return load(
        name="sparsevllm_shadowkv_host_gather_v5",
        sources=[str(source)],
        extra_include_paths=cuda_include_paths,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


__all__ = ["load_shadowkv_host_gather"]
