import math
import torch
from .abstract_attention import AbstractAttention
from .abstract_attention import AttentionUtils


class DenseAttention(AbstractAttention):
    """Standard dense attention with causal masking."""
    
    def __init__(self):
        super().__init__()

    def __call__(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        return AttentionUtils.flash_attention(queries, keys, values)


class BlockSparseAttentionMInference(AbstractAttention):
    """Block-sparse attention with chunk-level relevance and local windows."""
    
    def __init__(self, chunk_size: int, top_chunks: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.top_chunks = top_chunks
        
        assert chunk_size >= 8, "Recommended chunk size is >= 8"
        assert top_chunks >= 3, "Must select at least one top chunk"

    @staticmethod
    def _calculate_sparsity(seq_len: int, chunk_size: int, top_chunks: int) -> float:
        """Calculate sparsity ratio based on selected blocks vs total possible blocks.
        
        For autoregressive attention:
        - First query block picks 1 key block
        - First top_chunks query blocks pick max possible key blocks
        - Remaining query blocks pick top_chunks key blocks each
        
        Args:
            seq_len: Length of input sequence
            chunk_size: Size of each attention block
            top_chunks: Number of chunks to select per query
            
        Returns:
            Sparsity ratio between 0 and 1
        """
        num_blocks = (seq_len + chunk_size - 1) // chunk_size

        total_blocks = num_blocks * (num_blocks + 1) // 2
        
        selected_blocks = top_chunks * (top_chunks + 1) // 2
        selected_blocks += (num_blocks - top_chunks) * top_chunks
        
        return 1.0 - (selected_blocks / total_blocks)

    @staticmethod
    def _get_blocks_for_sparsity(seq_len: int, chunk_size: int, target_sparsity: float) -> int:
        """Calculate number of blocks needed to achieve desired sparsity level.
        
        Uses binary search to find the number of blocks that gives sparsity closest
        to the target. The relationship between blocks and sparsity is monotonic.
        
        Args:
            seq_len: Length of input sequence
            chunk_size: Size of each attention block
            target_sparsity: Desired sparsity ratio between 0 and 1
            
        Returns:
            Number of blocks to select per query to achieve target sparsity
        """
        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        
        # Binary search for number of blocks
        left, right = 3, num_blocks  # Minimum 4 blocks needed
        best_blocks = 3
        best_diff = float('inf')
        
        while left <= right:
            mid = (left + right) // 2
            sparsity = BlockSparseAttentionMInference._calculate_sparsity(seq_len, chunk_size, mid)
            
            diff = abs(sparsity - target_sparsity)
            if diff < best_diff:
                best_diff = diff
                best_blocks = mid
                
            if sparsity < target_sparsity:
                right = mid - 1
            else:
                left = mid + 1
                
        return best_blocks

    def __call__(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        from .minference import block_sparse_attention
        from sparse_frontier.modelling.attention.registry import get_attention_handler

        sparsity = float(self._calculate_sparsity(queries.shape[2], self.chunk_size, self.top_chunks))
        get_attention_handler().report_prefill_sparsity(sparsity)
        return block_sparse_attention(queries, keys, values, self.top_chunks, self.chunk_size, self.chunk_size)


class VerticalAndSlashAttentionMInference(AbstractAttention):
    """Combines vertical and diagonal patterns for efficient sparse attention.
    
    Implements the Vertical and Slash attention mechanism that selects important tokens based on:
    1) Vertical patterns - Top-k tokens that receive high attention across all queries
    2) Diagonal patterns - Diagonal stripes that capture local dependencies
    """

    def __init__(
        self,
        vertical_size: int = 64,
        slash_size: int = 128,
        approximation_size: int = 64,
    ):
        """Initialize Vertical and Slash attention.

        Args:
            vertical_size (int): Number of vertical tokens to select
            slash_size (int): Number of diagonal stripes to select
        """
        super().__init__()
        self.vertical_size = vertical_size
        self.slash_size = slash_size
        self.approximation_size = approximation_size

    def __call__(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        from .minference import vertical_and_slash_kernel
        from sparse_frontier.modelling.attention.registry import get_attention_handler

        attn_output, sparsity = vertical_and_slash_kernel(
            queries,
            keys,
            values,
            vertical_size=min(self.vertical_size, queries.shape[2]),
            slash_size=min(self.slash_size, queries.shape[2]),
            last_q=min(self.approximation_size, queries.shape[2]),
        )
        get_attention_handler().report_prefill_sparsity(float(sparsity))
        return attn_output


class FlexPrefill(AbstractAttention):
    def __init__(
        self,
        alpha: float = 0.9,
        approximation_size: int = 512,
        min_budget: int = 256,
    ):
        super().__init__()
        self.alpha = alpha
        self.approximation_size = approximation_size
        self.min_budget = min_budget
        self.causal_mask = None

    @staticmethod
    def score_cover_topk(x: torch.Tensor, score: float):
        cumsum_x = torch.cumsum(torch.sort(x, dim=-1, descending=True).values, dim=-1)
        topk = torch.sum(cumsum_x <= score, dim=-1) + 1
        return topk

    def get_active_blocks(
        self, q, k, v
    ):
        from .minference import sum_over_diagonals
        _, _, seq_len, head_dim = q.shape

        # Compute attention scores for last queries
        last_q_tokens = q[..., -self.approximation_size:, :] / math.sqrt(head_dim)
        qk = torch.einsum('bhik,bhjk->bhij', last_q_tokens, k)

        # Apply causal masking
        if self.causal_mask is None:
            self.causal_mask = torch.arange(0, self.approximation_size, device=last_q_tokens.device)
            self.causal_mask = self.causal_mask[:, None] >= self.causal_mask[None, :]
            self.causal_mask = self.causal_mask[None, None, ...]
        
        qk[..., -self.approximation_size:] = torch.where(
            self.causal_mask,
            qk[..., -self.approximation_size:],
            torch.tensor(float("-inf"), device=qk.device, dtype=qk.dtype)
        )

        # Get attention patterns
        scores = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)

        # Compute vertical patterns
        vertical = scores.mean(dim=-2)
        vertical_size = max(self.min_budget, self.score_cover_topk(vertical, self.alpha).item())
        vertical[..., :4] = float("inf")
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices
        
        # Fixed local window for slash patterns
        slashes = sum_over_diagonals(scores)[..., :-self.approximation_size + 1] / self.approximation_size
        slash_size = max(self.min_budget, self.score_cover_topk(slashes, self.alpha).item())
        slashes[..., -64:] = float("inf")
        slash = (seq_len - 1) - torch.topk(slashes, slash_size, -1).indices

        return vertical_topk, slash

    def __call__(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        from .minference import vertical_slash_sparse_attention
        from sparse_frontier.modelling.attention.registry import get_attention_handler

        assert queries.shape[-1] == keys.shape[-1]

        # Get active blocks for sparse portion
        vertical_idx, slash_idx = self.get_active_blocks(queries, keys, values)

        # Calculate sparse attention for non-dense portion
        sparse_out, sparsity = vertical_slash_sparse_attention(
            queries,
            keys,
            values,
            vertical_idx,
            slash_idx,
        )
        get_attention_handler().report_prefill_sparsity(float(sparsity))
        
        return sparse_out
