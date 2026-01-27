import math
import torch
import torch.nn.functional as F
from .abstract_attention import AbstractAttention


class SnapKVCompression(AbstractAttention):
    """SnapKV compression for efficient decoding"""
    def __init__(
        self,
        token_capacity: int,
        approximation_window: int = 256,
        kernel_size: int = 7,
        local_window: int = 128,
        prefix_tokens: int = 4,
    ):
        super().__init__()
        self.token_capacity = token_capacity
        self.approximation_window = approximation_window
        self.kernel_size = kernel_size
        self.local_window = local_window
        self.prefix_tokens = prefix_tokens
        self.causal_mask = None

    def kv_compress(
        self,
        queries: torch.Tensor,  # [num_tokens, num_q_heads, head_size]
        keys: torch.Tensor,     # [num_tokens, num_kv_heads, head_size]
        values: torch.Tensor,   # [num_tokens, num_kv_heads, head_size]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress KV cache using SnapKV approach with GQA support"""
        # Use last approximation_window queries to estimate importance
        approx_queries = queries[-self.approximation_window:] / math.sqrt(queries.size(-1))  # [approx_window, num_q_heads, head_size]

        num_q_heads = queries.size(1)
        num_kv_heads = keys.size(1)
        group_size = num_q_heads // num_kv_heads
        
        # Repeat keys for each query head in the group
        # [num_tokens, num_kv_heads, head_size] -> [num_tokens, num_q_heads, head_size]
        repeated_keys = keys.repeat_interleave(group_size, dim=1)
        
        # Now we can calculate attention scores with matching head dimensions
        scores = torch.einsum(
            'thd,nhd->htn', 
            approx_queries, 
            repeated_keys
        )
        
        # Initialize and cache causal mask if not already created
        if self.causal_mask is None:
            self.causal_mask = torch.arange(0, self.approximation_window, device=scores.device)
            self.causal_mask = self.causal_mask[:, None] >= self.causal_mask[None, :]
            self.causal_mask = self.causal_mask[None, ...]  # Add head dimension
        
        # Apply causal masking and softmax
        scores[..., -self.approximation_window:] = torch.where(
            self.causal_mask,
            scores[..., -self.approximation_window:],
            torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype)
        )

        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(keys.dtype)
        
        # Reshape attention weights to group query heads [num_kv_heads, group_size, approx_window, num_tokens]
        grouped_weights = attn_weights.view(num_kv_heads, group_size, -1, attn_weights.size(-1))
        # Average across group dimension [num_kv_heads, approx_window, num_tokens]
        token_importance = grouped_weights.mean(dim=1).sum(dim=1)  # [num_kv_heads, num_tokens]
        
        # Apply pooling for smoother selection per head
        token_importance = F.avg_pool1d(
            token_importance.unsqueeze(1),
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1
        ).squeeze(1)
        
        token_importance[..., -self.local_window:] = float('inf')
        token_importance[..., :self.prefix_tokens] = float('inf')

        # Select top-k tokens per head
        capacity = min(self.token_capacity, keys.size(0))

        assert capacity > self.local_window + self.prefix_tokens, f"Capacity {capacity} must be greater than local_window {self.local_window} + prefix_tokens {self.prefix_tokens}"

        _, indices = torch.topk(token_importance, k=capacity, dim=-1)  # [num_kv_heads, capacity]
        
        # Expand indices for gathering
        # [num_kv_heads, capacity] -> [num_kv_heads, capacity, head_size]
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, keys.size(-1))
        
        compressed_keys = torch.gather(
            keys.transpose(0, 1),  # [num_kv_heads, num_tokens, head_size]
            dim=1,
            index=expanded_indices
        )
        
        compressed_values = torch.gather(
            values.transpose(0, 1),  # [num_kv_heads, num_tokens, head_size]
            dim=1,
            index=expanded_indices
        )

        # Create sequence length tensor (same for all heads)
        seq_lens = torch.full((num_kv_heads,), capacity, device=queries.device, dtype=torch.long)
        
        return compressed_keys, compressed_values, seq_lens


class AdaSnapKVCompression(AbstractAttention):
    """Adaptive SnapKV compression with non-uniform token distribution across heads"""
    def __init__(
        self,
        token_capacity: int,
        approximation_window: int = 256,
        kernel_size: int = 7,
        local_window: int = 128,
        prefix_tokens: int = 4,
        min_head_capacity_ratio: float = 0.2,
    ):
        super().__init__()
        self.token_capacity = token_capacity
        self.approximation_window = approximation_window
        self.kernel_size = kernel_size
        self.local_window = local_window
        self.prefix_tokens = prefix_tokens
        self.min_head_capacity_ratio = min_head_capacity_ratio
        self.causal_mask = None

    def kv_compress(
        self,
        queries: torch.Tensor,  # [num_tokens, num_q_heads, head_size]
        keys: torch.Tensor,     # [num_tokens, num_kv_heads, head_size]
        values: torch.Tensor,   # [num_tokens, num_kv_heads, head_size]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress KV cache using AdaSnapKV approach with adaptive token distribution"""
        assert self.approximation_window < keys.size(0)

        # Use last approximation_window queries to estimate importance
        approx_queries = queries[-self.approximation_window:] / math.sqrt(queries.size(-1))  # [approx_window, num_q_heads, head_size]

        num_q_heads = queries.size(1)
        num_kv_heads = keys.size(1)
        group_size = num_q_heads // num_kv_heads
        
        # Repeat keys for each query head in the group
        # [num_tokens, num_kv_heads, head_size] -> [num_tokens, num_q_heads, head_size]
        repeated_keys = keys.repeat_interleave(group_size, dim=1)
        
        # Now we can calculate attention scores with matching head dimensions
        scores = torch.einsum(
            'thd,nhd->htn', 
            approx_queries, 
            repeated_keys
        )
        
        # Initialize and cache causal mask if not already created
        if self.causal_mask is None:
            self.causal_mask = torch.arange(0, self.approximation_window, device=scores.device)
            self.causal_mask = self.causal_mask[:, None] >= self.causal_mask[None, :]
            self.causal_mask = self.causal_mask[None, ...]  # Add head dimension
        
        # Apply causal masking and softmax
        scores[..., -self.approximation_window:] = torch.where(
            self.causal_mask,
            scores[..., -self.approximation_window:],
            torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype)
        )

        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(keys.dtype)
        
        grouped_weights = attn_weights.view(num_kv_heads, group_size, -1, attn_weights.size(-1))

        token_importance = grouped_weights.max(dim=2)[0].max(dim=1)[0]
        
        # Apply pooling for smoother selection per head
        token_importance = F.avg_pool1d(
            token_importance.unsqueeze(1),
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1
        ).squeeze(1)

        # Always keep recent tokens and prefix tokens
        token_importance[..., -self.local_window:] = float('inf')
        token_importance[..., :self.prefix_tokens] = float('inf')

        # Calculate total capacity and minimum capacity per head
        total_capacity = self.token_capacity * num_kv_heads
        assert total_capacity <= keys.size(0) * num_kv_heads
        min_capacity_per_head = int(self.token_capacity * self.min_head_capacity_ratio)
        min_capacity_per_head = max(min_capacity_per_head, self.local_window + self.prefix_tokens)
        assert self.token_capacity >= self.local_window + self.prefix_tokens
        assert min_capacity_per_head <= keys.size(0)
        assert total_capacity >= min_capacity_per_head * num_kv_heads
        remaining_capacity = total_capacity - (min_capacity_per_head * num_kv_heads)
        
        # Get the top-k tokens for minimum capacity per head
        _, min_indices = torch.topk(token_importance, k=min_capacity_per_head, dim=-1)  # [num_kv_heads, min_capacity]
        
        # Vectorized mask creation using scatter
        selected_mask = torch.zeros_like(token_importance, dtype=torch.bool)
        selected_mask.scatter_(
            dim=1,
            index=min_indices,
            src=torch.ones_like(min_indices, dtype=torch.bool)
        )

        # Vectorized masking
        masked_importance = token_importance.masked_fill(selected_mask, float('-inf'))
        flat_importance = masked_importance.view(-1)
        
        # Global selection with vectorized index conversion
        _, flat_indices = torch.topk(flat_importance, k=remaining_capacity, dim=-1)
        
        # Flatten and update selected_mask
        flat_selected_mask = selected_mask.view(-1)
        flat_selected_mask.scatter_(0, flat_indices, True)
        selected_mask = flat_selected_mask.view(num_kv_heads, -1)
        
        seq_lens = selected_mask.sum(dim=1)
        max_seq_len = seq_lens.max().item()
        compressed_keys = torch.zeros(num_kv_heads, max_seq_len, keys.size(-1), device=keys.device, dtype=keys.dtype)
        compressed_values = torch.zeros(num_kv_heads, max_seq_len, values.size(-1), device=values.device, dtype=values.dtype)
        
        keys_t = keys.transpose(0, 1)
        values_t = values.transpose(0, 1)
        
        for head_idx in range(num_kv_heads):
            compressed_keys[head_idx, :seq_lens[head_idx]] = keys_t[head_idx, selected_mask[head_idx]]
            compressed_values[head_idx, :seq_lens[head_idx]] = values_t[head_idx, selected_mask[head_idx]]
        
        return compressed_keys, compressed_values, seq_lens
