import math
import torch
import torch.distributed as dist
import torch.nn.functional as F

from flash_attn import flash_attn_varlen_func
from flash_attn import flash_attn_with_kvcache

from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.registry import register_ops
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple


__all__ = [
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "prefill_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
    "silu_and_mul",
    "linear",
]


@register_ops(vendor_ops_registry)
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    new_states = hidden_states + residual
    residual = new_states
    output = rms_norm(new_states, weight, epsilon)
    return output, residual


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    half_size = x.shape[-1] // 2
    x1 = x[..., :half_size]
    x2 = x[..., half_size:]
    out = torch.empty_like(x)
    out[..., :half_size] = -x2
    out[..., half_size:] = x1
    return out


@register_ops(vendor_ops_registry)
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    position_ids: Optional[Tensor],
    cos_sin_cache: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    "not inplace"
    query = query.contiguous()
    key = key.contiguous()
    q_embed = (query * cos) + (rotate_half(query) * sin)
    k_embed = (key * cos) + (rotate_half(key) * sin)
    return q_embed, k_embed


@register_ops(vendor_ops_registry)
def prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    max_q_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    if q_seq_len is None:
        q_seq_len = max_q_seq_len
    max_kv_seq_len = max_q_seq_len
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    causal = True
    if softmax_scale is None:
        softmax_scale = float(1 / math.sqrt(key.size(-1)))
    output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=q_start_loc,
        cu_seqlens_k=q_start_loc,
        max_seqlen_q=max_q_seq_len,
        max_seqlen_k=max_kv_seq_len,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
    )
    return output


@register_ops(vendor_ops_registry)
def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    kv_indices = kv_indices.flatten()
    key = key.contiguous()
    value = value.contiguous()
    _, block_size, _, _ = key_cache.shape
    for i in range(kv_indices.size(0)):
        slot_idx = kv_indices[i]
        block_idx = slot_idx // block_size
        block_offset = slot_idx % block_size
        key_cache[block_idx, block_offset] = key[i]
        value_cache[block_idx, block_offset] = value[i]
    return key_cache, value_cache


@register_ops(vendor_ops_registry)
def paged_decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Optional[Tensor],
    block_size: int,
    kv_seq_len: Tensor,
    max_kv_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError("paged_decode_attention does not support alibi_slopes yet")
    query = query.contiguous()
    dim = query.size(-1)
    batch_size = block_table.size(0)
    if softmax_scale is None:
        softmax_scale = float(1 / math.sqrt(query.size(-1)))

    block_table = block_table.to(torch.int32)
    kv_seq_len = kv_seq_len.to(torch.int32).to(query.device)
    output = flash_attn_with_kvcache(
        query.view(batch_size, -1, num_q_heads, dim),
        key_cache,
        value_cache,
        cache_seqlens=kv_seq_len,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=True,
    )
    return output


@register_ops(vendor_ops_registry)
def paged_prefill_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    block_size: int,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    dim = query.size(-1)
    batch_size = block_table.size(0)

    if softmax_scale is None:
        softmax_scale = float(1 / math.sqrt(query.size(-1)))
    output = flash_attn_with_kvcache(
        query.view(batch_size, -1, num_q_heads, dim),
        key_cache,
        value_cache,
        cache_seqlens=kv_seq_len.to(torch.int32).to(query.device),
        block_table=block_table.to(torch.int32),
        softmax_scale=softmax_scale,
        causal=True,
    )
    return output


@register_ops(vendor_ops_registry)
def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    hidden_states = hidden_states.contiguous()
    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = weight * x.to(input_dtype)
    return x


@register_ops(vendor_ops_registry)
def silu_and_mul(x: Tensor, dim: int = -1) -> Tensor:
    gate, up = x.chunk(2, dim)
    output = F.silu(gate) * up
    return output


@register_ops(vendor_ops_registry)
def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    all_reduce: Optional[bool],
) -> Tensor:
    out = torch.nn.functional.linear(x, weight, bias)
    if all_reduce:
        dist.all_reduce(out)
    return out
