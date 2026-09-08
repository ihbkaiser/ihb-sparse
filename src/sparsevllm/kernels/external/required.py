from __future__ import annotations

from sparsevllm.kernels.external.flashinfer.support import (
    flashinfer_kernel_health,
    flashinfer_kernel_metadata_health,
)
from sparsevllm.kernels.external.sgl.support import (
    sgl_kernel_health,
    sgl_kernel_metadata_health,
)
from sparsevllm.kernels.external.support import (
    CUDA_DEPENDENCY_INSTALL_HINT,
    KernelFamilyHealth,
)


def _raise_for_unhealthy(
    health_by_family: tuple[KernelFamilyHealth, ...],
    *,
    stage: str,
) -> None:
    unhealthy = tuple(health for health in health_by_family if not health.ready)
    if not unhealthy:
        return
    details = "; ".join(
        f"{health.family} is {health.state.value}: {health.reason}"
        for health in unhealthy
    )
    families = " and ".join(health.family for health in health_by_family)
    raise RuntimeError(
        "Sparse-vLLM CUDA engine requires healthy "
        f"{families} dependencies during {stage}, but {details}. "
        f"{CUDA_DEPENDENCY_INSTALL_HINT}"
    )


def config_requires_sgl_kernel(config: object) -> bool:
    """Return whether the configured method has a hard SGL dependency.

    Quest and ShadowKV have repository-owned CUDA/Triton or FlashInfer
    providers and therefore treat SGL FA3 as an optional portfolio member.
    Keep the historical strict requirement for all other methods until their
    provider dependency contracts are made equally explicit.
    """
    sparse_method = str(getattr(config, "sparse_method", "") or "").strip().lower()
    return sparse_method not in {"quest", "shadowkv"}


def validate_required_cuda_kernel_metadata(*, require_sgl: bool = True) -> None:
    """Validate required packages before workers start without importing them."""

    health = [flashinfer_kernel_metadata_health()]
    if require_sgl:
        health.append(sgl_kernel_metadata_health())
    _raise_for_unhealthy(tuple(health), stage="startup metadata validation")


def validate_required_cuda_kernel_families(*, require_sgl: bool = True) -> None:
    """Import required kernel packages after the rank's CUDA device is selected."""

    health = [flashinfer_kernel_health()]
    if require_sgl:
        health.append(sgl_kernel_health())
    _raise_for_unhealthy(tuple(health), stage="device-bound binary validation")


__all__ = [
    "validate_required_cuda_kernel_families",
    "validate_required_cuda_kernel_metadata",
    "config_requires_sgl_kernel",
]
