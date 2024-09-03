# Copyright (c) 2024, DeepLink. All rights reserved.
import math
import torch
import torch_npu

from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.registry import register_ops
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple

import json
from dataclasses import dataclass, asdict
torch.classes.load_library("/data2/weitao/atb_models/output/atb_speed/lib/libatb_speed_torch.so")

def create_op(op_name: str, params: Optional[str] = None, out = False):
    op = torch.classes.OperationTorch.OperationTorch(op_name)
    if params is not None:
        params = json.dumps(asdict(params))
        op.set_param(params)
    else:
        params = json.dumps({})
        op.set_param(params)
    if out:
        return op.execute_out_with_param
    else:
        return op.execute 
    

__all__ = [
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "prefill_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
    "moe_gating_topk_softmax",
    "get_cache_len",
]


@register_ops(vendor_ops_registry)
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    normed_hidden_states, _, added_hidden_states = torch.ops.npu.npu_add_rms_norm(
        hidden_states, residual, weight, epsilon
    )
    return normed_hidden_states, added_hidden_states


@dataclass
class RotaryParams:
    rotaryCoeff: int = 2


@register_ops(vendor_ops_registry)
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    position_ids: Optional[Tensor],
    cos_sin_cache: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    if len(cos.shape) < 4:
        cos = cos.unsqueeze(2)
    if len(sin.shape) < 4:
        sin = sin.unsqueeze(2)
    query = query.contiguous()
    key = key.contiguous()
    torch.ops.npu.npu_apply_rotary_pos_emb(query, key, cos, sin, "BSND")
    return query, key
    # bsz, seq_len, num_q_heads, head_dim = query.shape
    # _, _, num_kv_heads, _ = key.shape
    # query = query.view(bsz * seq_len, num_q_heads * head_dim)
    # key = key.view(bsz * seq_len, num_kv_heads * head_dim)
    # cos = cos.view(bsz * seq_len, head_dim)
    # sin = sin.view(bsz * seq_len, head_dim)
    # seq_len = torch.tensor([seq_len], dtype=torch.int32).cuda()
    # params = RotaryParams(rotaryCoeff = 2)
    # execute = create_op("RopeOperation", params)
    # ropeQ_atb, ropeK_atb = execute([query, key, cos, sin, seq_len])
    # ropeQ_atb = ropeQ_atb.view(bsz, seq_len, num_q_heads, head_dim)
    # ropeK_atb = ropeK_atb.view(bsz, seq_len, num_kv_heads, head_dim)
    # return ropeQ_atb, ropeK_atb


@dataclass 
class PagedAttentionPrefillParams:
    headNum: int
    kvHeadNum: int
    qkScale: float = 1.0
    qScale: float = 1.0
    calcType: int = 3
    kernelType: int = 1
    clampType: int = 0
    isTriuMask: int = 1
    maskType: int = 0


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
    if alibi_slopes is not None:
        raise RuntimeError(
            "paged_decode_attention does not " "support alibi_slopes yet"
        )
    # cann prompt_fa don't support batch query with different seq_len
    seq_len_list = None if q_seq_len is None else q_seq_len.tolist()

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    if attn_mask is not None:
        params = PagedAttentionPrefillParams(
        headNum=num_q_heads,
        kvHeadNum=num_kv_heads,
        qkScale=1.0 / math.sqrt(query.shape[-1]),
        qScale=1.0, 
        calcType=3, #paged encode
        kernelType=0, # 1为高精度，0为默认
        isTriuMask=1, #triu mask
        maskType=1, #norm mask
)    
        execute = create_op("SelfAttentionOperation", params, True)
        query = query.view(-1, num_q_heads * query.shape[-1])
        key = key.view(-1, num_kv_heads * key.shape[-1])
        value = value.view(-1, num_kv_heads * value.shape[-1])
        # mask要求f16/bf16, seq_len要求int32
        execute([query, key, value, attn_mask[0].type(torch.float16), q_seq_len.type(torch.int32)],[attn_output], json.dumps({ "seqLen": seq_len_list,"hasMask": True,}))
    else:
        params = PagedAttentionPrefillParams(
        headNum=num_q_heads,
        kvHeadNum=num_kv_heads,
        qkScale=1.0 / math.sqrt(query.shape[-1]),
        qScale=1.0, 
        calcType=3, #paged encode
        maskType=0, #norm mask
        )    
        execute = create_op("SelfAttentionOperation", params, True)
        execute([query, key, value],[attn_output], json.dumps({ "seqLen": seq_len_list, "hasMask": False,}))
    return attn_output


@register_ops(vendor_ops_registry)
def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    head, dim = key.shape[1:]
    block_num, block_size = key_cache.shape[:2]
    block_total = block_num * block_size

    # only support contiguous k,v
    key = key.contiguous()
    value = value.contiguous()

    key_cache_reshaped = key_cache.view(block_total, head, dim)
    value_cache_reshaped = value_cache.view(block_total, head, dim)
    torch.ops.npu.npu_scatter_nd_update_(key_cache_reshaped, kv_indices, key)
    torch.ops.npu.npu_scatter_nd_update_(value_cache_reshaped, kv_indices, value)
    return key_cache, value_cache

def fill_kv_cache_atb(key, value, key_cache, value_cache, kv_indices):
    kv_indices = kv_indices.flatten()
    execute = create_op("ReshapeAndCacheOperation")
    updated_k_cache, updated_v_cache = execute([key, value, key_cache, value_cache, kv_indices])
    return updated_k_cache, updated_v_cache


@register_ops(vendor_ops_registry)
def fill_contiguous_kvcache(
    key_cache: Tensor, value_cache: Tensor, key_state: Tensor, value_state: Tensor
) -> Tuple[Tensor, Tensor]:
    key_cache = torch.cat([key_cache, key_state], dim=1)
    value_cache = torch.cat([value_cache, value_state], dim=1)
    return key_cache, value_cache


@register_ops(vendor_ops_registry)
def get_cache_len(cache: Tensor):
    return cache.shape[1]

@dataclass
class PagedAttentionParams:
    headNum: int
    qkScale: float
    kvHeadNum: int
    maskType: int = 0

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
        raise RuntimeError(
            "paged_decode_attention does not " "support alibi_slopes yet"
        )
    if isinstance(block_table, torch.Tensor) and block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)

    ntokens, _, head_size = query.shape
    total_block, _, _ = key_cache.shape
    key_cache = key_cache.view(total_block//block_size, block_size, num_kv_heads, head_size)
    value_cache = value_cache.view(total_block//block_size, block_size, num_kv_heads, head_size)
    contextLens = kv_seq_len.tolist()
    params = PagedAttentionParams(headNum=num_q_heads, qkScale=1.0 / math.sqrt(head_size), kvHeadNum=num_kv_heads)
    execute = create_op("PagedAttentionOperation", params, True)
    execute([query, key_cache, value_cache, block_table.type(torch.int32), kv_seq_len.to(torch.int32)], [attn_output], json.dumps({"contextLen":contextLens}))
    return attn_output


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
    if alibi_slopes is not None:
        raise RuntimeError(
            "paged_decode_attention does not " "support alibi_slopes yet"
        )
    if softmax_scale is not None:
        raise RuntimeError(
            "paged_decode_attention does not " "support softmax_scale yet"
        )
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)

    # cann incre_fa don't support paged_attn when q_seq_len > 1
    batch = q_start_loc.shape[0]
    q_seq_len_list = q_seq_len.tolist()
    kv_seq_len_list = kv_seq_len.tolist()
    scale_value = 1.0 / math.sqrt(query.shape[-1])
    query = query.contiguous()
    for i in range(batch):
        start = q_start_loc[i]
        mask = attn_mask[i]
        for j in range(q_seq_len_list[i]):
            single_q = query[start + j : start + j + 1].view(1, 1, -1)
            single_o = attn_output[start + j : start + j + 1].view(1, 1, -1)
            torch.ops.npu_ext.npu_incre_flash_attention_v4_out(
                single_q,
                key_cache,
                value_cache,
                single_o,
                padding_mask=None,
                atten_mask=mask[j : j + 1],
                actual_seq_lengths=kv_seq_len_list[i : i + 1],
                antiquant_scale=None,
                antiquant_offset=None,
                block_table=block_table,
                dequant_scale1=None,
                quant_scale1=None,
                dequant_scale2=None,
                quant_scale2=None,
                quant_offset2=None,
                num_heads=num_q_heads,
                scale_value=scale_value,
                input_layout="BSH",
                num_key_value_heads=num_kv_heads,
                block_size=block_size,
                inner_precise=1,
            )
    return attn_output


@dataclass
class RmsNormParams:
    epsilon: float


@register_ops(vendor_ops_registry)
def rms_norm(hidden_states: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    params = RmsNormParams(epsilon=epsilon)
    execute = create_op("RmsNormOperation", params)
    return execute([hidden_states, weight])[0]


@register_ops(vendor_ops_registry)
def moe_gating_topk_softmax(router_logits: Tensor, topk: int) -> Tuple[Tensor, Tensor]:
    routing_weights = router_logits.new_empty((*router_logits.shape[:-1], topk))
    selected_experts = router_logits.new_empty(
        (*router_logits.shape[:-1], topk), dtype=torch.int32
    )
    selected_idx = torch.empty_like(selected_experts)
    return torch.ops.npu_ext.npu_moe_gating_topk_softmax(
        router_logits, None, topk, routing_weights, selected_experts, selected_idx
    )


# TODO only for internlm on transformers lib.
# see issue #9 for details
@register_ops(vendor_ops_registry)
def fused_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    mask: Sequence[Optional[Tensor]],
) -> Tensor:
    batch_size = query_states.shape[0]
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    q_seq_len, num_q_heads, _ = query_states.shape
    kv_seq_len, num_kv_heads, _ = value_states.shape
    attn_output = torch.empty_like(query_states)

    for i in range(batch_size):
        if q_seq_len == kv_seq_len:
            prefill_attention(
                query_states,
                key_states,
                value_states,
                torch.tensor(
                    [kv_seq_len - q_seq_len],
                    dtype=torch.int64,
                    device=query_states.device,
                ),
                torch.tensor(
                    [kv_seq_len], dtype=torch.int64, device=query_states.device
                ),
                num_q_heads,
                num_kv_heads,
                mask[i : i + 1],
                None,
                None,
                attn_output,
            )
        else:
            paged_decode_attention(
                query_states,
                key_states,
                value_states,
                None,
                0,
                torch.tensor(
                    [kv_seq_len], dtype=torch.int64, device=query_states.device
                ),
                num_q_heads,
                num_kv_heads,
                None,
                None,
                attn_output,
            )
    return attn_output
