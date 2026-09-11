from tools.query_robust.final_calibration import allocate_stratified_queries


def test_allocate_stratified_queries_preserves_total_and_bucket_balance():
    """A 3K query-head budget must not expand to 3K per position bucket."""

    allocation = allocate_stratified_queries(
        total_samples=3000,
        num_sequences=20,
        buckets=("8k", "16k", "24k", "32k"),
    )

    assert sum(allocation.values()) == 3000
    assert {
        bucket: sum(
            count for (sequence, label), count in allocation.items() if label == bucket
        )
        for bucket in ("8k", "16k", "24k", "32k")
    } == {"8k": 750, "16k": 750, "24k": 750, "32k": 750}
    assert {
        sum(
            count
            for (sequence_index, _), count in allocation.items()
            if sequence_index == sequence
        )
        for sequence in range(20)
    } == {148, 152}
