from __future__ import annotations

from .passthrough import PassThroughRuntime
from .base import SparseStepContext


class ShadowKVRuntime(PassThroughRuntime):
    """ShadowKV has cache-owned selection and no controller score workspace."""

    def finish_step(self, step: SparseStepContext) -> None:
        if not step.is_prefill:
            return
        if not any(seq.is_last_chunk_prefill for seq in step.seqs):
            return
        finalize = getattr(self.cache_manager, "finalize_shadowkv_prefill", None)
        if finalize is None:
            raise RuntimeError("ShadowKV runtime is not paired with its cache manager.")
        finalize(step.seqs)


__all__ = ["ShadowKVRuntime"]
