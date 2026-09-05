import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
import torch.nn.functional as F

from sparse_frontier.modelling.attention import shadowkv
from sparse_frontier.modelling.attention.shadowkv import ShadowKVAttention


class ScalarRope:
    """A deterministic scalar-position RoPE stand-in for algorithm tests."""

    head_size = 4

    def __init__(self):
        self.calls = []

    def forward(self, positions, query, key):
        positions = positions.flatten().to(key.device, dtype=key.dtype)
        key = key.view(positions.numel(), -1, self.head_size).clone()
        key[..., 0] += positions[:, None]
        key[..., 1] -= positions[:, None]
        self.calls.append(positions.detach().clone())
        return query, key.flatten(1)


def _rope_key(rope, positions, key):
    if positions.ndim == 2:
        rotated = []
        for head in range(key.shape[0]):
            query = torch.zeros_like(key[head])
            _, result = rope.forward(positions[head], query, key[head])
            rotated.append(result.view(-1, key.shape[-1]))
        return torch.stack(rotated)
    flat = key.transpose(0, 1).reshape(positions.numel(), -1)
    query = torch.zeros(positions.numel(), rope.head_size, dtype=flat.dtype)
    _, result = rope.forward(positions, query, flat)
    return result.view(positions.numel(), key.shape[0], key.shape[-1]).transpose(0, 1)


def _official_prefill_reference(
    pre_rope,
    post_rope,
    rank,
    sparse_budget,
    chunk_size,
    local_chunk,
    outlier_chunk,
):
    num_tokens, num_kv_heads, head_dim = pre_rope.shape
    flat = pre_rope.reshape(1, num_tokens, -1)
    u, s, v = torch.svd(flat.float())
    u = u[:, :, :rank].squeeze(0).to(pre_rope.dtype)
    sv = torch.matmul(torch.diag_embed(s[:, :rank]), v.transpose(1, 2)[:, :rank])[0]
    sv = sv.view(rank, num_kv_heads, head_dim).permute(1, 0, 2).to(pre_rope.dtype)

    chunks = num_tokens // chunk_size - local_chunk
    ctx = post_rope[:chunks * chunk_size]
    ctx = ctx.transpose(0, 1).reshape(num_kv_heads, chunks, chunk_size, head_dim)
    landmarks = ctx.mean(dim=-2)
    cos = F.cosine_similarity(
        landmarks.unsqueeze(2).expand(-1, -1, chunk_size, -1),
        ctx,
        dim=-1,
    )
    outlier = cos.min(dim=-1).values.topk(outlier_chunk, largest=False).indices
    all_idx = torch.arange(chunks).unsqueeze(0).expand(num_kv_heads, -1)
    mask = torch.ones_like(all_idx, dtype=torch.bool)
    mask.scatter_(1, outlier, False)
    rest = all_idx.masked_select(mask).view(num_kv_heads, -1)
    landmarks = landmarks.gather(1, rest.unsqueeze(-1).expand(-1, -1, head_dim))
    return u, sv, landmarks, rest, outlier


def _reference_selected(queries, landmarks, landmark_indices, groups, select_sets, chunk_size):
    _, num_q_heads, head_dim = queries.shape
    num_kv_heads = landmarks.shape[0]
    q = queries.view(1, num_kv_heads, groups, 1, head_dim)
    scores = torch.einsum("bhgqd,bhdc->bhgqc", q, landmarks.unsqueeze(0).transpose(2, 3))
    scores = scores.squeeze(2)
    scores = torch.softmax(scores / math.sqrt(128), dim=-1, dtype=torch.float32)
    scores = scores.sum(dim=-2)
    if groups > 1:
        scores = scores.max(dim=-2).values
    chosen = torch.topk(scores, k=select_sets, dim=-1).indices
    selected_chunks = landmark_indices.unsqueeze(0).gather(-1, chosen).squeeze(0)
    positions = (
        selected_chunks.unsqueeze(-1) * chunk_size
        + torch.arange(chunk_size).view(1, 1, -1)
    ).reshape(num_kv_heads, -1)
    return selected_chunks, positions


def _read_temp_cache(k_cache, v_cache, block_table, cache_seqlens, num_q_heads):
    keys = []
    values = []
    for head in range(num_q_heads):
        length = int(cache_seqlens[head].item())
        blocks = block_table[head]
        head_keys = []
        head_values = []
        for block_idx, block in enumerate(blocks):
            take = min(k_cache.shape[1], length - block_idx * k_cache.shape[1])
            if take <= 0:
                break
            head_keys.append(k_cache[block, :take, 0, :])
            head_values.append(v_cache[block, :take, 0, :])
        keys.append(torch.cat(head_keys))
        values.append(torch.cat(head_values))
    return torch.stack(keys), torch.stack(values)


def test_shadowkv_matches_official_prefill_and_retrieval(monkeypatch):
    torch.manual_seed(7)
    tokens, kv_heads, groups, head_dim = 12, 2, 2, 4
    q_heads = kv_heads * groups
    config = dict(
        sparse_budget=4,
        chunk_size=2,
        rank=3,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        tp_size=1,
        block_size=4,
    )
    attention = ShadowKVAttention(**config)
    pre_rope = torch.randn(tokens, kv_heads, head_dim)
    post_rope = torch.randn_like(pre_rope)
    values = torch.randn_like(pre_rope)
    positions = torch.arange(tokens)
    rope = ScalarRope()

    expected = _official_prefill_reference(
        pre_rope,
        post_rope,
        rank=3,
        sparse_budget=4,
        chunk_size=2,
        local_chunk=1,
        outlier_chunk=1,
    )
    attention.prefill_state(
        pre_rope,
        post_rope,
        values,
        positions,
        layer_idx=0,
        rope_forward=rope.forward,
    )

    expected_u, expected_sv, expected_landmarks, expected_landmark_idx, expected_outlier = expected
    torch.testing.assert_close(attention.U[0, 0], expected_u)
    torch.testing.assert_close(attention.SV[0, 0], expected_sv)
    torch.testing.assert_close(attention.k_landmark[0, 0], expected_landmarks)
    torch.testing.assert_close(attention.k_landmark_idx[0, 0], expected_landmark_idx)
    torch.testing.assert_close(attention.outlier_chunk_idx[0, 0], expected_outlier)

    query = torch.randn(1, q_heads, head_dim)
    chosen, selected_positions = _reference_selected(
        query,
        expected_landmarks,
        expected_landmark_idx,
        groups,
        select_sets=2,
        chunk_size=2,
    )
    actual_chosen, actual_positions = attention.select_chunks(query, layer_idx=0)
    torch.testing.assert_close(actual_chosen[0], chosen)
    torch.testing.assert_close(actual_positions[0], selected_positions)


def test_shadowkv_prefill_delegates_to_dense_attention(monkeypatch):
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=2,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        tp_size=1,
        block_size=4,
    )
    query = torch.randn(1, 4, 8, 4)
    key = torch.randn(1, 2, 8, 4)
    value = torch.randn_like(key)
    expected = torch.randn_like(query)
    seen = {}

    def dense_reference(q, k, v):
        seen["inputs"] = (q, k, v)
        return expected

    monkeypatch.setattr(shadowkv.AttentionUtils, "flash_attention", dense_reference)
    actual = attention(query, key, value, layer_idx=0)
    assert actual is expected
    assert seen["inputs"][0] is query
    assert seen["inputs"][1] is key
    assert seen["inputs"][2] is value


def test_shadowkv_reconstruction_and_flashattention_order(monkeypatch):
    torch.manual_seed(11)
    tokens, kv_heads, groups, head_dim = 12, 2, 2, 4
    q_heads = kv_heads * groups
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=3,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        tp_size=1,
        block_size=4,
    )
    pre_rope = torch.randn(tokens, kv_heads, head_dim)
    post_rope = torch.randn_like(pre_rope)
    values = torch.randn_like(pre_rope)
    rope = ScalarRope()
    attention.prefill_state(
        pre_rope,
        post_rope,
        values,
        torch.arange(tokens),
        layer_idx=0,
        rope_forward=rope.forward,
    )

    current_k = torch.randn(1, kv_heads, head_dim)
    current_v = torch.randn_like(current_k)
    cache_k = torch.zeros(kv_heads, 4, 4, head_dim)
    cache_v = torch.zeros_like(cache_k)
    cache_k.view(kv_heads, -1, head_dim)[:, :tokens] = post_rope.transpose(0, 1)
    cache_v.view(kv_heads, -1, head_dim)[:, :tokens] = values.transpose(0, 1)
    cache_k.view(kv_heads, -1, head_dim)[:, tokens] = current_k[0]
    cache_v.view(kv_heads, -1, head_dim)[:, tokens] = current_v[0]
    tokens_per_head = torch.full((q_heads,), tokens + 1, dtype=torch.int32)
    query = torch.randn(1, q_heads, head_dim)
    output = torch.empty_like(query)
    observed = {}

    def fake_flash(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        fa_version,
        out,
    ):
        del cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, fa_version
        observed["keys"], observed["values"] = k.transpose(0, 1), v.transpose(0, 1)
        q_heads_local = q.shape[1]
        result = []
        for head in range(q_heads_local):
            score = observed["keys"][head // groups] @ q[0, head] / math.sqrt(head_dim)
            weight = torch.softmax(score.float(), dim=-1).to(score.dtype)
            result.append(weight @ observed["values"][head // groups])
        out.copy_(torch.stack(result).unsqueeze(0))

    monkeypatch.setattr(shadowkv, "flash_attn_varlen_func", fake_flash)
    attention.decode(
        query,
        current_k,
        current_v,
        cache_k,
        cache_v,
        tokens_per_head,
        output,
        layer_idx=0,
    )

    selected_chunks, selected_positions = attention.select_chunks(query, 0)
    del selected_chunks
    local_start = tokens - attention.prefill_local
    local_positions = torch.arange(local_start, tokens)
    outlier_chunks = attention.outlier_chunk_idx[0].to(torch.long)
    outlier_positions = (
        outlier_chunks.unsqueeze(-1) * attention.chunk_size
        + torch.arange(attention.chunk_size).view(1, 1, -1)
    ).view(kv_heads, -1)
    generated_positions = torch.tensor([tokens])
    selected_pre = torch.gather(
        attention.U[0, 0].unsqueeze(0).expand(kv_heads, -1, -1),
        1,
        selected_positions[0].unsqueeze(-1).expand(-1, -1, attention.rank),
    )
    selected_pre = torch.einsum("knr,k r d->knd", selected_pre, attention.SV[0, 0])
    selected_post = _rope_key(rope, selected_positions[0], selected_pre)
    expected_k = torch.cat(
        [
            cache_k.view(kv_heads, -1, head_dim)[:, local_positions],
            cache_k.view(kv_heads, -1, head_dim).gather(
                1, outlier_positions.unsqueeze(-1).expand(-1, -1, head_dim)
            ),
            selected_post,
            cache_k.view(kv_heads, -1, head_dim)[:, generated_positions],
        ],
        dim=1,
    )
    expected_v = torch.cat(
        [
            cache_v.view(kv_heads, -1, head_dim)[:, local_positions],
            cache_v.view(kv_heads, -1, head_dim).gather(
                1, outlier_positions.unsqueeze(-1).expand(-1, -1, head_dim)
            ),
            cache_v.view(kv_heads, -1, head_dim).gather(
                1, selected_positions[0].unsqueeze(-1).expand(-1, -1, head_dim)
            ),
            cache_v.view(kv_heads, -1, head_dim)[:, generated_positions],
        ],
        dim=1,
    )
    torch.testing.assert_close(observed["keys"], expected_k)
    torch.testing.assert_close(observed["values"], expected_v)
    for head in range(q_heads):
        score = expected_k[head // groups] @ query[0, head] / math.sqrt(head_dim)
        expected = torch.softmax(score.float(), dim=-1) @ expected_v[head // groups]
        torch.testing.assert_close(output[0, head], expected)


def test_shadowkv_global_svd_gathers_heads_and_keeps_local_sv(monkeypatch):
    torch.manual_seed(19)
    local = torch.randn(10, 2, 3)
    remote = torch.randn(10, 2, 3)
    gathered = torch.cat([local, remote], dim=1)

    monkeypatch.setattr(
        shadowkv,
        "tensor_model_parallel_all_gather",
        lambda tensor, dim=-1: torch.cat([tensor, remote], dim=dim),
    )
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=2,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=8,
        num_kv_heads=4,
        tp_size=2,
        block_size=4,
        tp_rank=0,
    )
    post = torch.randn_like(local)
    attention.prefill_state(
        local,
        post,
        torch.randn_like(local),
        torch.arange(local.shape[0]),
        layer_idx=0,
        rope_forward=ScalarRope().forward,
    )
    u, s, v = torch.svd(
        gathered.reshape(1, gathered.shape[0], -1).float()
    )
    expected_u = u[0, :, :2].to(local.dtype)
    expected_sv = s[0, :2, None] * v[0][:, :2].transpose(-1, -2)
    expected_sv = expected_sv.view(2, 4, 3).permute(1, 0, 2).to(local.dtype)[:2]
    torch.testing.assert_close(attention.U[0, 0], expected_u)
    torch.testing.assert_close(attention.SV[0, 0], expected_sv)
    assert attention.SV.shape[2] == 2


def test_shadowkv_randomized_backend_requests_truncated_svd(monkeypatch):
    calls = {}

    def fake_svd_lowrank(matrix, q, niter):
        calls["q"] = q
        calls["niter"] = niter
        u, s, v = torch.svd(matrix)
        return u[:, :, :q], s[:, :q], v[:, :, :q]

    monkeypatch.setattr(shadowkv.torch, "svd_lowrank", fake_svd_lowrank)
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=3,
        svd_backend="randomized",
        svd_oversample=2,
        svd_niter=2,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        tp_size=1,
        block_size=4,
        fused_retrieval=True,
    )
    pre_rope = torch.randn(12, 2, 4)
    attention.prefill_state(
        pre_rope,
        torch.randn_like(pre_rope),
        torch.randn_like(pre_rope),
        torch.arange(12),
        layer_idx=0,
        rope_forward=ScalarRope().forward,
    )

    assert calls == {"q": 5, "niter": 2}
    assert attention.U.shape[-1] == 3
    assert attention.SV.shape[-2] == 3


def test_shadowkv_gesvdj_backend_uses_full_pre_rope_svd(monkeypatch):
    calls = {}
    u = torch.randn(1, 5, 3)
    s = torch.randn(1, 3)
    vh = torch.randn(1, 3, 3)

    def fake_svd(matrix, full_matrices, driver):
        calls.update(shape=matrix.shape, full_matrices=full_matrices, driver=driver)
        return u, s, vh

    monkeypatch.setattr(shadowkv.torch.linalg, "svd", fake_svd)
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=3,
        svd_backend="gesvdj",
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        tp_size=1,
        block_size=4,
    )

    actual_u, actual_s, actual_v = attention._compute_svd(torch.randn(1, 5, 3))

    assert calls == {"shape": torch.Size([1, 5, 3]), "full_matrices": False, "driver": "gesvdj"}
    torch.testing.assert_close(actual_u, u)
    torch.testing.assert_close(actual_s, s)
    torch.testing.assert_close(actual_v, vh.transpose(-2, -1))


def test_shadowkv_gesvdj_falls_back_to_official_svd_on_driver_failure(monkeypatch):
    calls = []

    def failed_svd(*args, **kwargs):
        calls.append("gesvdj")
        raise RuntimeError("cuSOLVER convergence failure")

    expected = tuple(torch.randn(1, 5, 3) for _ in range(3))

    def official_svd(matrix):
        calls.append("torch.svd")
        return expected

    monkeypatch.setattr(shadowkv.torch.linalg, "svd", failed_svd)
    monkeypatch.setattr(shadowkv.torch, "svd", official_svd)
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=3,
        svd_backend="gesvdj",
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        tp_size=1,
        block_size=4,
    )

    actual = attention._compute_svd(torch.randn(1, 5, 3))

    assert calls == ["gesvdj", "torch.svd"]
    assert actual is expected


def test_shadowkv_retrieval_handles_single_query_per_kv_group():
    torch.manual_seed(23)
    tokens, kv_heads, head_dim = 12, 2, 4
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=3,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=kv_heads,
        num_kv_heads=kv_heads,
        tp_size=1,
        block_size=4,
    )
    pre_rope = torch.randn(tokens, kv_heads, head_dim)
    attention.prefill_state(
        pre_rope,
        torch.randn_like(pre_rope),
        torch.randn_like(pre_rope),
        torch.arange(tokens),
        layer_idx=0,
        rope_forward=ScalarRope().forward,
    )
    query = torch.randn(1, kv_heads, head_dim)
    chosen, positions = attention.select_chunks(query, layer_idx=0)
    assert chosen.shape == (1, kv_heads, 2)
    assert positions.shape == (1, kv_heads, 4)


def test_shadowkv_batches_selected_rope_dispatch_across_kv_heads():
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=3,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        tp_size=1,
        block_size=4,
    )
    pre_rope = torch.randn(12, 2, 4)
    rope = ScalarRope()
    attention.prefill_state(
        pre_rope,
        torch.randn_like(pre_rope),
        torch.randn_like(pre_rope),
        torch.arange(12),
        layer_idx=0,
        rope_forward=rope.forward,
    )
    rope.calls.clear()
    selected_pre_rope = torch.randn(2, 3, 4)
    selected_positions = torch.tensor([[1, 4, 7], [2, 5, 8]])
    actual = attention._apply_rope_per_head(selected_pre_rope, selected_positions, 0)
    assert len(rope.calls) == 1
    expected = _rope_key(rope, selected_positions, selected_pre_rope)
    torch.testing.assert_close(actual, expected)


def test_shadowkv_uses_direct_varlen_flash_for_assembled_tokens(monkeypatch):
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=3,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        tp_size=1,
        block_size=4,
    )
    query = torch.randn(1, 4, 4)
    keys = torch.randn(6, 2, 4)
    values = torch.randn_like(keys)
    output = torch.empty_like(query)
    observed = {}

    def fake_flash(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        fa_version,
        out,
    ):
        observed.update(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            fa_version=fa_version,
        )
        out.fill_(3)
        return out

    monkeypatch.setattr(shadowkv, "flash_attn_varlen_func", fake_flash)
    actual = attention._flash_attention_assembled(query, keys, values, output)

    assert actual is output
    assert observed["q"].shape == (1, 4, 4)
    assert observed["k"].shape == (6, 2, 4)
    assert observed["v"].shape == (6, 2, 4)
    torch.testing.assert_close(observed["q"], query)
    torch.testing.assert_close(observed["k"], keys)
    torch.testing.assert_close(observed["v"], values)
    torch.testing.assert_close(
        observed["cu_seqlens_q"], torch.tensor([0, 1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        observed["cu_seqlens_k"], torch.tensor([0, 6], dtype=torch.int32)
    )
    assert observed["max_seqlen_q"] == 1
    assert observed["max_seqlen_k"] == 6
    assert observed["causal"] is True
    assert observed["fa_version"] == 2
    torch.testing.assert_close(output, torch.full_like(output, 3))


def test_shadowkv_reuses_decode_assembly_buffers_without_changing_order():
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=3,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        tp_size=1,
        block_size=4,
    )
    cache_k = torch.randn(2, 16, 4)
    cache_v = torch.randn_like(cache_k)
    positions = (
        torch.tensor([0, 1]),
        torch.tensor([[2, 4], [3, 5]]),
        torch.tensor([[6, 8], [7, 9]]),
        torch.tensor([10, 11]),
    )
    selected_keys = torch.randn(2, 2, 4)

    first_keys, first_values = attention._assemble_decode_from_cache(
        cache_k, cache_v, positions, selected_keys
    )
    first_key_ptr = first_keys.untyped_storage().data_ptr()
    first_value_ptr = first_values.untyped_storage().data_ptr()
    second_keys, second_values = attention._assemble_decode_from_cache(
        cache_k, cache_v, positions, selected_keys
    )

    expected_keys = torch.cat(
        [
            torch.stack([cache_k[0, positions[0]], cache_k[1, positions[0]]], dim=1),
            torch.stack([cache_k[0, positions[1][0]], cache_k[1, positions[1][1]]], dim=1),
            selected_keys.transpose(0, 1),
            torch.stack([cache_k[0, positions[3]], cache_k[1, positions[3]]], dim=1),
        ],
        dim=0,
    )
    expected_values = torch.cat(
        [
            torch.stack([cache_v[0, positions[0]], cache_v[1, positions[0]]], dim=1),
            torch.stack([cache_v[0, positions[1][0]], cache_v[1, positions[1][1]]], dim=1),
            torch.stack([cache_v[0, positions[2][0]], cache_v[1, positions[2][1]]], dim=1),
            torch.stack([cache_v[0, positions[3]], cache_v[1, positions[3]]], dim=1),
        ],
        dim=0,
    )
    torch.testing.assert_close(second_keys, expected_keys)
    torch.testing.assert_close(second_values, expected_values)
    assert second_keys.untyped_storage().data_ptr() == first_key_ptr
    assert second_values.untyped_storage().data_ptr() == first_value_ptr


def test_shadowkv_gathers_cache_directly_into_token_major_layout():
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=3,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        tp_size=1,
        block_size=4,
    )
    cache = torch.arange(2 * 6 * 4, dtype=torch.float32).view(2, 6, 4)
    positions = torch.tensor([[0, 4, 1], [2, 5, 3]])
    destination = torch.empty(3, 2, 4)

    attention._copy_cache_tokens(cache, positions, destination)

    expected = torch.stack(
        [cache[0, positions[0]], cache[1, positions[1]]], dim=1
    )
    torch.testing.assert_close(destination, expected)


def test_shadowkv_uses_fused_gather_gemm_rope_when_available(monkeypatch):
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=2,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        tp_size=1,
        block_size=4,
    )
    attention.U = torch.randn(1, 1, 10, 2, dtype=torch.bfloat16)
    attention.SV = torch.randn(1, 1, 2, 2, 4, dtype=torch.bfloat16)
    attention._rope_kernel_metadata[0] = {
        "cos_sin_cache": torch.randn(32, 4, dtype=torch.bfloat16),
        "head_size": 4,
        "rotary_dim": 4,
        "is_neox_style": True,
    }
    positions = torch.tensor([[1, 5], [2, 6]], dtype=torch.long)
    expected = torch.randn(2, 2, 4, dtype=torch.bfloat16)
    observed = {}

    class FakeCuda:
        def gather_gemm_rope(self, u, sv, position_ids, cos_sin_cache):
            observed.update(
                u=u,
                sv=sv,
                position_ids=position_ids,
                cos_sin_cache=cos_sin_cache,
            )
            return expected

    monkeypatch.setattr(shadowkv, "_shadowkv_cuda", FakeCuda())
    actual = attention._reconstruct_selected_keys(positions, layer_idx=0)

    assert actual is expected
    torch.testing.assert_close(observed["u"], attention.U[0, 0])
    torch.testing.assert_close(observed["sv"], attention.SV[0, 0])
    torch.testing.assert_close(observed["position_ids"], positions)
    assert observed["cos_sin_cache"] is attention._rope_kernel_metadata[0]["cos_sin_cache"]


def test_shadowkv_uses_fused_landmark_softmax_when_available(monkeypatch):
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=2,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        tp_size=1,
        block_size=4,
        fused_retrieval=True,
    )
    attention.k_landmark = torch.randn(1, 1, 2, 5, 4, dtype=torch.bfloat16)
    attention.k_landmark_idx = torch.arange(10).view(1, 1, 2, 5)
    attention.initialized = True
    query = torch.randn(1, 4, 4, dtype=torch.bfloat16)
    expected_scores = torch.zeros(1, 2, 2, 1, 5)
    expected_scores[:, :, :, :, 3] = 1
    expected_scores[:, :, :, :, 0] = 0.5
    observed = {}

    class FakeCuda:
        def retrieval_softmax(self, q, landmarks):
            observed.update(q=q, landmarks=landmarks)
            return expected_scores

    monkeypatch.setattr(shadowkv, "_shadowkv_cuda", FakeCuda())
    chosen, _ = attention.select_chunks(query, layer_idx=0)

    torch.testing.assert_close(observed["q"], query)
    torch.testing.assert_close(observed["landmarks"], attention.k_landmark[0])
    torch.testing.assert_close(chosen, torch.tensor([[[3, 0], [8, 5]]]))


def test_shadowkv_uses_reference_landmark_softmax_by_default():
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=2,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        tp_size=1,
        block_size=4,
    )
    assert attention.fused_retrieval is False


def test_shadowkv_cuda_retrieval_kernel_matches_reference():
    if not torch.cuda.is_available() or shadowkv._shadowkv_cuda is None:
        pytest.skip("optional ShadowKV CUDA extension is unavailable")

    torch.manual_seed(23)
    query = torch.randn(1, 4, 4, device="cuda", dtype=torch.bfloat16)
    landmarks = torch.randn(1, 2, 7, 4, device="cuda", dtype=torch.bfloat16)
    actual = shadowkv._shadowkv_cuda.retrieval_softmax(query, landmarks)
    scores = torch.einsum(
        "bhgd,bhcd->bhgc",
        query.view(1, 2, 2, 4).float(),
        landmarks.float(),
    ) / math.sqrt(128)
    expected = torch.softmax(scores, dim=-1).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shadowkv_cuda_gather_gemm_rope_matches_reference():
    if not torch.cuda.is_available() or shadowkv._shadowkv_cuda is None:
        pytest.skip("optional ShadowKV CUDA extension is unavailable")

    torch.manual_seed(29)
    heads, tokens, selected, rank, head_dim = 2, 11, 5, 3, 4
    u = torch.randn(tokens, rank, device="cuda", dtype=torch.bfloat16)
    sv = torch.randn(heads, rank, head_dim, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor(
        [[0, 3, 5, 8, 10], [1, 4, 6, 9, 2]], device="cuda", dtype=torch.long
    )
    cos_sin = torch.randn(16, head_dim, device="cuda", dtype=torch.bfloat16)
    actual = shadowkv._shadowkv_cuda.gather_gemm_rope(u, sv, positions, cos_sin)

    pre_rope = torch.einsum("hsr,hrd->hsd", u[positions].float(), sv.float())
    cos = cos_sin[positions][..., : head_dim // 2].float()
    sin = cos_sin[positions][..., head_dim // 2 :].float()
    expected = torch.cat(
        [
            pre_rope[..., : head_dim // 2] * cos
            - pre_rope[..., head_dim // 2 :] * sin,
            pre_rope[..., head_dim // 2 :] * cos
            + pre_rope[..., : head_dim // 2] * sin,
        ],
        dim=-1,
    ).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sparse_budget": 0},
        {"chunk_size": 0},
        {"rank": 0},
        {"sparse_budget": 5, "chunk_size": 2},
    ],
)
def test_shadowkv_rejects_invalid_configuration(kwargs):
    defaults = dict(
        sparse_budget=4,
        chunk_size=2,
        rank=2,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        tp_size=1,
        block_size=4,
    )
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        ShadowKVAttention(**defaults)


def test_shadowkv_rejects_short_prompt_and_reset_clears_state():
    attention = ShadowKVAttention(
        sparse_budget=4,
        chunk_size=2,
        rank=2,
        local_chunk=1,
        outlier_chunk=1,
        num_layers=1,
        num_q_heads=4,
        num_kv_heads=2,
        tp_size=1,
        block_size=4,
    )
    short = torch.randn(7, 2, 3)
    with pytest.raises(ValueError, match="short|minimum"):
        attention.prefill_state(
            short,
            short.clone(),
            short.clone(),
            torch.arange(short.shape[0]),
            layer_idx=0,
            rope_forward=ScalarRope().forward,
        )

    long = torch.randn(10, 2, 3)
    attention.prefill_state(
        long,
        long.clone(),
        long.clone(),
        torch.arange(long.shape[0]),
        layer_idx=0,
        rope_forward=ScalarRope().forward,
    )
    assert attention.U is not None
    attention.reset()
    assert not attention.initialized
    assert attention.U is None
    assert attention.SV is None
    assert attention.k_landmark is None
    assert attention.k_landmark_idx is None
    assert attention.outlier_chunk_idx is None
    assert attention.selected_chunk_idx is None
