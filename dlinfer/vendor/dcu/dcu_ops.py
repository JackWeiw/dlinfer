# Copyright (c) 2024, DeepLink. All rights reserved.
import math
import torch

from flash_attn import flash_attn_varlen_func
from flash_attn import flash_attn_with_kvcache

from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.registry import register_ops
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple

from .maca_extension import ops as maca_ext_ops

__all__ = [
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "prefill_attention",
    "fused_moe",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
    "silu_and_mul",
    "moe_gating_topk_softmax",
]


@register_ops(vendor_ops_registry)
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    maca_ext_ops.fused_add_rms_norm(hidden_states, residual, weight, epsilon)
    return hidden_states, residual


@register_ops(vendor_ops_registry)
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    position_ids: Optional[Tensor],
    cos_sin_cache: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    position_ids_1d = torch.arange(0, query.size(1), device=query.device)
    query = query.flatten(-2, -1)
    key = key.flatten(-2, -1)
    cos = cos.squeeze(0).squeeze(1)
    cos = cos[..., : cos.shape[-1] // 2]
    sin = sin.squeeze(0).squeeze(1)
    sin = sin[..., : sin.shape[-1] // 2]
    cos_sin_cache = torch.cat((cos, sin), dim=-1)

    maca_ext_ops.rotary_embedding(
        position_ids_1d, query, key, cos_sin_cache.size(-1), cos_sin_cache, True
    )
    return query, key