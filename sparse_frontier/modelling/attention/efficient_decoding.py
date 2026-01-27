import torch
from typing import Optional, Tuple, List
from .abstract_attention import AbstractAttention
from .abstract_attention import AttentionUtils
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_with_kvcache


def _update_last_page(
    page_reps: torch.Tensor,
    keys: torch.Tensor,  # [1, num_heads, head_dim]
    tokens_per_head_int: int,
    page_size: int,
):
    """Update representations of the page containing the current token.
    
    Args:
        page_reps: Page representations tensor
        keys: Key tensor for the current token
        tokens_per_head_int: Number of tokens (as int)
        page_size: Size of each page in the KV cache
    """
    current_page_idx: int = (tokens_per_head_int - 1) // page_size
    page_reps[current_page_idx, 0] = torch.minimum(
        page_reps[current_page_idx, 0],
        keys.squeeze(0)
    )
    
    page_reps[current_page_idx, 1] = torch.maximum(
        page_reps[current_page_idx, 1],
        keys.squeeze(0)
    )


def _select_pages(
    page_reps: torch.Tensor,
    query: torch.Tensor,
    tokens_per_head_int: int,
    page_size: int,
    page_budget: int,
    offsets: torch.Tensor,
    share_pages: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select most relevant pages based on query-page similarity.
    
    Args:
        page_reps: Page representations tensor
        query: Query tensor for the current token
        tokens_per_head_int: Number of tokens (as int)
        page_size: Size of each page in the KV cache
        page_budget: Maximum number of pages to select
        offsets: Offsets for each KV head
        share_pages: Whether to share KV pages across Query heads
        
    Returns:
        Tuple of (selected page indices, new tokens per head)
    """
    current_page_idx = (tokens_per_head_int - 1) // page_size
    num_q_heads = query.shape[1]
    num_kv_heads = page_reps.size(2)
    group_size = num_q_heads // num_kv_heads

    # Dense case: not enough pages to require sparse selection
    if current_page_idx <= page_budget:
        indices = torch.arange(page_budget + 1, device=query.device, dtype=torch.int32)
        indices = torch.clamp(indices, max=current_page_idx)
        indices = indices.unsqueeze(0).expand(num_q_heads, -1)
        new_tokens_per_head = torch.full((num_q_heads,), tokens_per_head_int, device=query.device, dtype=torch.int32)
        return indices.int() + offsets, new_tokens_per_head
    
    query_squeezed = query.squeeze(0) # We assume query is [1, num_heads, head_dim]
    page_reps = page_reps[:current_page_idx].repeat_interleave(group_size, dim=2)
        
    scores = torch.einsum(
        'hd,prhd->hprd',
        query_squeezed,
        page_reps
    )

    scores = scores.max(dim=2).values  # [num_heads, num_pages, head_dim]

    if share_pages:
        scores = scores.reshape(num_kv_heads, group_size, scores.shape[1], -1)
        scores = scores.sum(dim=(1, 3))
    else:
        scores = scores.sum(dim=-1)
    
    _, indices = torch.topk(
        scores,
        k=page_budget + 1,
        dim=1,
        sorted=True,
    )

    indices[:, -1] = current_page_idx
    new_tokens_per_head = tokens_per_head_int - (current_page_idx - page_budget) * page_size
    new_tokens_per_head = torch.full((query.shape[1],), new_tokens_per_head, device=query.device, dtype=torch.int32)

    if share_pages:
        indices = indices.repeat_interleave(group_size, dim=0)

    active_pages = indices.int() + offsets
    return active_pages, new_tokens_per_head


_select_pages = torch.compile(_select_pages)
_update_last_page = torch.jit.script(_update_last_page)


class QuestAttention(AbstractAttention):
    """Quest attention for efficient decoding with dynamic page selection.
    
    Quest maintains min and max representations for each page of KV cache and uses
    them to dynamically select the most relevant pages during decoding.
    """
    
    def __init__(
        self,
        token_budget: int,
        page_size: int,
        max_input_tokens: int,
        max_output_tokens: int,
        num_layers: int,
        share_pages: bool,
    ):
        """Initialize Quest attention.
        
        Args:
            token_budget: Maximum number of tokens to attend to
            page_size: Size of each page in the KV cache
            max_input_tokens: Maximum input token length (from config)
            max_output_tokens: Maximum output token length (from config)
            num_layers: Number of transformer layers (from model config)
            share_pages: Whether to share pages across query heads
        """
        super().__init__()
        self.token_budget = token_budget
        self.page_size = page_size
        self.page_budget = token_budget // page_size
        assert token_budget % page_size == 0, "Token budget must be divisible by page size"

        self.share_pages = share_pages
        
        self.max_pages = ((max_input_tokens + max_output_tokens) + page_size - 1) // page_size
        self.num_layers = num_layers
        
        # Page representations per layer
        self.page_reps_per_layer: List[Optional[torch.Tensor]] = [None] * num_layers
        self.offsets = None

    def preallocate_memory(self, keys: torch.Tensor) -> None:
        """Pre-allocate page representations for all layers."""
        num_heads = keys.shape[1]
        head_dim = keys.shape[-1]
        
        for layer_idx in range(self.num_layers):
            if self.page_reps_per_layer[layer_idx] is None:
                self.page_reps_per_layer[layer_idx] = torch.zeros(
                    self.max_pages, 2, num_heads, head_dim,
                    device=keys.device,
                    dtype=keys.dtype
                )
        
    def _init_page_reps(
        self,
        keys: torch.Tensor,
        layer_idx: int = 0,
    ):
        """Initialize page representations during prefilling for a specific layer."""
        _, num_heads, seq_len, head_dim = keys.shape
        keys = keys.squeeze(0).transpose(0, 1)  # [seq_len, num_heads, head_dim]
        
        num_pages = (seq_len + self.page_size - 1) // self.page_size

        self.page_reps_per_layer[layer_idx][:, 0] = float('inf')
        self.page_reps_per_layer[layer_idx][:, 1] = float('-inf')

        if seq_len % self.page_size != 0:
            complete_pages = seq_len // self.page_size
            complete_keys = keys[:complete_pages * self.page_size].view(complete_pages, self.page_size, num_heads, head_dim)
            complete_min = complete_keys.amin(dim=1)
            complete_max = complete_keys.amax(dim=1)
            self.page_reps_per_layer[layer_idx][:complete_pages] = torch.stack([complete_min, complete_max], dim=1)
            
            remainder_keys = keys[complete_pages * self.page_size:]
            
            self.page_reps_per_layer[layer_idx][complete_pages] = torch.stack([
                remainder_keys.amin(dim=0),
                remainder_keys.amax(dim=0),
            ], dim=0)
        else:
            keys = keys.view(num_pages, self.page_size, num_heads, head_dim)
            self.page_reps_per_layer[layer_idx][:num_pages] = torch.stack([
                keys.amin(dim=1),
                keys.amax(dim=1),
            ], dim=1)
        
    def __call__(
        self,
        queries: torch.Tensor,  # [batch_size, num_heads, seq_len, head_dim]
        keys: torch.Tensor,     # [batch_size, num_kv_heads, seq_len, head_dim]
        values: torch.Tensor,   # [batch_size, num_kv_heads, seq_len, head_dim]
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            queries: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            keys: Key tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
            values: Value tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
            layer_idx: Index of the current transformer layer
            
        Returns:
            Attention output tensor of shape [batch_size, num_heads, seq_len, head_dim]
        """
        self._init_page_reps(keys, layer_idx)

        return AttentionUtils.flash_attention(queries, keys, values)
        
    def decode(
        self,
        query: torch.Tensor,  # [1, num_heads, head_dim]
        keys: torch.Tensor,   # [1, num_kv_heads, head_dim]
        values: torch.Tensor, # [1, num_kv_heads, head_dim]
        k_cache: torch.Tensor,  # [num_kv_heads, num_blocks, block_size, head_dim]
        v_cache: torch.Tensor,  # [num_kv_heads, num_blocks, block_size, head_dim]
        tokens_per_head: torch.Tensor,  # [num_heads]
        output: torch.Tensor, # [1, num_heads, head_dim]
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Compute attention during decoding with dynamic page selection for a specific layer.
        
        Instead of retrieving the selected KV cache, we pass the entire KV cache
        and the selected page indices to flash_attn_with_kvcache.
        
        Args:
            query: Query tensor for a single token [1, num_heads, head_dim]
            keys: Key tensor for the current token [1, num_kv_heads, head_dim]
            values: Value tensor for the current token [1, num_kv_heads, head_dim]
            k_cache: Key cache tensor [num_kv_heads, num_blocks, block_size, head_dim]
            v_cache: Value cache tensor [num_kv_heads, num_blocks, block_size, head_dim]
            tokens_per_head: Tensor of sequence lengths per head [num_heads]
            output: Output tensor to store results [1, num_heads, head_dim]
            layer_idx: Index of the current transformer layer
            
        Returns:
            Attention output tensor of shape [1, num_heads, head_dim]
        """
        num_kv_heads, num_blocks, block_size, head_size = k_cache.shape
        _, num_q_heads, _ = query.shape

        if self.offsets is None:
            offsets = torch.arange(num_kv_heads, device=query.device, dtype=torch.int32)
            offsets = offsets.repeat_interleave(num_q_heads // num_kv_heads)
            offsets = offsets.unsqueeze(1) * num_blocks
            self.offsets = offsets

        tokens_per_head_int = tokens_per_head[0].item()

        _update_last_page(
            page_reps=self.page_reps_per_layer[layer_idx],
            keys=keys,
            tokens_per_head_int=tokens_per_head_int,
            page_size=self.page_size
        )
        
        active_pages, new_tokens_per_head = _select_pages(
            page_reps=self.page_reps_per_layer[layer_idx],
            query=query,
            tokens_per_head_int=tokens_per_head_int,
            page_size=self.page_size,
            page_budget=self.page_budget,
            offsets=self.offsets,
            share_pages=self.share_pages,
        )

        flash_attn_with_kvcache(
            q=query.squeeze(0).unsqueeze(1).unsqueeze(1),
            k_cache=k_cache.view(num_kv_heads * num_blocks, block_size, 1, head_size),
            v_cache=v_cache.view(num_kv_heads * num_blocks, block_size, 1, head_size),
            block_table=active_pages,
            cache_seqlens=new_tokens_per_head,
            causal=True,
            out=output.squeeze(0).unsqueeze(1).unsqueeze(1),
        )

class TOVAAttention(AbstractAttention):
    def __init__(
        self,
        token_budget: int,
    ):
        super().__init__()
        self.token_budget = token_budget
    
    def __call__(
        self,
        queries: torch.Tensor,  # [batch_size, num_heads, seq_len, head_dim]
        keys: torch.Tensor,     # [batch_size, num_kv_heads, seq_len, head_dim]
        values: torch.Tensor,   # [batch_size, num_kv_heads, seq_len, head_dim]
        layer_idx: int = 0,
    ) -> torch.Tensor:
        return AttentionUtils.flash_attention(queries, keys, values)
    
    def compress(
        self,
        query: torch.Tensor,  # [1, num_q_heads, head_size]
        k_cache: torch.Tensor,     # [num_kv_heads, num_blocks, block_size, head_size]
        v_cache: torch.Tensor,     # [num_kv_heads, num_blocks, block_size, head_size]
        tokens_per_head: torch.Tensor,  # [num_kv_heads]
    ):
        current_length = tokens_per_head[0].item()
        if current_length <= self.token_budget:
            return

        num_q_heads, num_kv_heads = query.shape[1], k_cache.shape[0]
        k_cache = k_cache.view(num_kv_heads, -1, k_cache.shape[-1])
        v_cache = v_cache.view(num_kv_heads, -1, v_cache.shape[-1])

        gqa_group_size = num_q_heads // num_kv_heads

        scores = torch.einsum(
            'hd,hnd->hn',
            query.squeeze(0),
            k_cache[:, :current_length].repeat_interleave(gqa_group_size, dim=0),
        )
        scores = scores.view(num_kv_heads, gqa_group_size, -1).sum(dim=1)

        if current_length == self.token_budget + 1:
            min_indices = scores.argmin(dim=-1)
            rows = torch.arange(num_kv_heads, device=query.device, dtype=torch.int32)
            k_cache[rows, min_indices] = k_cache[rows, current_length - 1]
            v_cache[rows, min_indices] = v_cache[rows, current_length - 1]
        else:
            _, indices = torch.topk(scores, k=self.token_budget, dim=-1)
            # [num_kv_heads, capacity] -> [num_kv_heads, capacity, head_size]
            expanded_indices = indices.unsqueeze(-1).expand(-1, -1, k_cache.size(-1))
            k_cache[:, :self.token_budget] = torch.gather(k_cache, dim=1, index=expanded_indices)
            v_cache[:, :self.token_budget] = torch.gather(v_cache, dim=1, index=expanded_indices)

        tokens_per_head.fill_(self.token_budget)

    def decode(
        self,
        query: torch.Tensor,           # [1, num_query_heads, head_dim]
        keys: torch.Tensor,            # [1, num_kv_heads, head_dim]  (current token)
        values: torch.Tensor,          # [1, num_kv_heads, head_dim]  (current token)
        k_cache: torch.Tensor,         # [num_kv_heads, num_blocks, block_size, head_dim]
        v_cache: torch.Tensor,         # [num_kv_heads, num_blocks, block_size, head_dim]
        tokens_per_head: torch.Tensor, # [num_kv_heads]
        output: torch.Tensor,          # [1, num_query_heads, head_dim]
        layer_idx: int = 0,
    ) -> torch.Tensor:
        self.compress(query, k_cache, v_cache, tokens_per_head)
        return super().decode(query, keys, values, k_cache, v_cache, tokens_per_head, output, layer_idx)
