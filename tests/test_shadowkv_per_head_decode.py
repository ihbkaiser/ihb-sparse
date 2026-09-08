import pytest
import torch

from sparsevllm.kernels.triton.shadowkv_per_head_decode import shadowkv_per_head_decode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shadowkv_per_head_decode_matches_independent_attention_oracle_and_graph():
    torch.manual_seed(20260907)
    batch, query_heads, kv_heads, width, head_dim = 2, 8, 2, 48, 32
    q = torch.randn(batch, query_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_heads, width, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    head_lens = torch.tensor(
        [[40, 33], [48, 29]], dtype=torch.int32, device="cuda"
    )
    max_splits = 4
    mid_o = torch.empty(
        batch, query_heads, max_splits, head_dim, device="cuda", dtype=torch.float32
    )
    mid_lse = torch.empty(
        batch, query_heads, max_splits, device="cuda", dtype=torch.float32
    )
    output = torch.empty_like(q)
    output_lse = torch.empty(query_heads, batch, device="cuda", dtype=torch.float32)
    shadowkv_per_head_decode(
        q,
        k,
        v,
        head_lens,
        mid_o,
        mid_lse,
        softmax_scale=head_dim**-0.5,
        target_tokens_per_split=16,
        block_n=16,
        num_warps=4,
        output=output,
        output_lse=output_lse,
    )
    reference = torch.empty_like(output, dtype=torch.float32)
    for batch_idx in range(batch):
        for query_head in range(query_heads):
            kv_head = query_head // (query_heads // kv_heads)
            length = int(head_lens[batch_idx, kv_head].item())
            logits = (
                q[batch_idx, query_head].float()
                @ k[batch_idx, kv_head, :length].float().transpose(0, 1)
            ) * (head_dim**-0.5)
            reference[batch_idx, query_head] = torch.softmax(logits, dim=-1) @ v[
                batch_idx, kv_head, :length
            ].float()
    torch.testing.assert_close(output.float(), reference, rtol=4e-2, atol=4e-2)

    graph_output = torch.empty_like(output)
    graph_lse = torch.empty_like(output_lse)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        shadowkv_per_head_decode(
            q,
            k,
            v,
            head_lens,
            mid_o,
            mid_lse,
            softmax_scale=head_dim**-0.5,
            target_tokens_per_split=16,
            block_n=16,
            num_warps=4,
            output=graph_output,
            output_lse=graph_lse,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output.float(), reference, rtol=4e-2, atol=4e-2)
