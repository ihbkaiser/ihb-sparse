"""Deterministic sampling utilities for frozen Query-Robust calibrations."""

from __future__ import annotations


def allocate_stratified_queries(
    *,
    total_samples: int,
    num_sequences: int,
    buckets: tuple[str, ...],
) -> dict[tuple[int, str], int]:
    """Allocate one query-head budget equally across position buckets.

    Remainders go to the earliest sequence indices so the result is stable and
    every bucket has exactly the same total count.
    """

    if total_samples <= 0 or num_sequences <= 0 or not buckets:
        raise ValueError("total_samples, num_sequences, and buckets must be positive.")
    if total_samples % len(buckets):
        raise ValueError("total_samples must divide evenly across buckets.")

    per_bucket = total_samples // len(buckets)
    per_sequence, remainder = divmod(per_bucket, num_sequences)
    return {
        (sequence_index, bucket): per_sequence + (sequence_index < remainder)
        for bucket in buckets
        for sequence_index in range(num_sequences)
    }
