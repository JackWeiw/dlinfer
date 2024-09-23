# Copyright (c) 2024, DeepLink. All rights reserved.
import math
import torch
import torch_npu

from typing import List, Dict
from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.registry import register_ops
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple

# from torch.profiler import record_function
import json
from dataclasses import dataclass, asdict
torch.classes.load_library("/data2/weitao/atb_models/output/atb_speed/lib/libatb_speed_torch.so")

atb_op_manager = {}

atb_op = torch.classes.OperationTorch.OperationTorch("ATB")

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
    "linear",
]

@dataclass
class LinearParams:
    transposeA: bool = False
    transposeB: bool = True 
    hasBias: bool = True
    outDataType: int = -1 # dtype undefined和输入一致
    name: str = "LinearOperation"

    
@register_ops(vendor_ops_registry)
def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    deqscale: Optional[Tensor],
    transpose_a: bool = False,
    transpose_b: bool = True,
) -> Tensor:
    if deqscale is not None:
        raise RuntimeError("linear does not support deqscale yet")
    has_bais = bias is not None
    if "LineraOperation" not in atb_op_manager:
        params = LinearParams(transposeA=transpose_a, transposeB=transpose_b, hasBias=has_bais)
        atb_op = torch.classes.OperationTorch.OperationTorch("LinearOperation")
        atb_op.set_param(json.dumps(params.__dict__))
        atb_op_manager["LinearOperation"] = atb_op
    if has_bais:
        import pdb; pdb.set_trace()
        return atb_op_manager["LinearOperation"].execute([x, weight, bias])[0]
    else:
        return atb_op_manager["LinearOperation"].execute([x, weight])[0]

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
    name: str = "RopeOperation"


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
            
    if "RopeOperation" not in atb_op_manager:
        params = RotaryParams(rotaryCoeff = 2)
        atb_op = torch.classes.OperationTorch.OperationTorch("RopeOperation")
        # with torch.profiler.record_function("set_param"):
        atb_op.set_param(json.dumps(params.__dict__))
        atb_op_manager["RopeOperation"] = atb_op
        
    bsz, seq_len, num_q_heads, head_dim = query.shape
    _, _, num_kv_heads, _ = key.shape
    query = query.view(bsz * seq_len, num_q_heads * head_dim)
    key = key.view(bsz * seq_len, num_kv_heads * head_dim)
    cos = cos.view(bsz * seq_len, head_dim)
    sin = sin.view(bsz * seq_len, head_dim)
    seq_len = torch.tensor([seq_len], dtype=torch.int32).cuda()

    # with torch.profiler.record_function("set_op_name"):
    #     atb_op.set_op_name("RopeOperation")
    # with torch.profiler.record_function("set_param"):
    #     atb_op.set_param(json.dumps(params.__dict__))
    # with torch.profiler.record_function("execute_out"):
    atb_op_manager["RopeOperation"].execute_out([query, key, cos, sin, seq_len], [query, key])
    ropeQ_atb = query.view(bsz, seq_len, num_q_heads, head_dim)
    ropeK_atb = key.view(bsz, seq_len, num_kv_heads, head_dim)
    return ropeQ_atb, ropeK_atb

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
    name: str = "SelfAttentionOperation"

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
    if attn_output is None:
        attn_output = torch.empty_like(query).cuda()
        import pdb; pdb.set_trace()

    attn_output = prefill_attention_atb(query, key, value, q_seq_len, num_q_heads, num_kv_heads, attn_mask, attn_output)
    return attn_output

def prefill_attention_atb(query, key, value, q_seq_len,  num_q_heads, num_kv_heads, attn_mask, attn_output):
    seq_len_list = [] if q_seq_len is None else q_seq_len.tolist()
    if "SelfAttentionOperation" not in atb_op_manager:
        atb_op = torch.classes.OperationTorch.OperationTorch("SelfAttentionOperation")
        params = PagedAttentionPrefillParams(
        headNum=num_q_heads,
        kvHeadNum=num_kv_heads,
        qkScale=1.0 / math.sqrt(query.shape[-1]),
        qScale=1.0, 
        calcType=3, #paged encode
        kernelType=1, # 1为高精度，0为默认
        isTriuMask=1, #triu mask
        maskType=1, #norm mask
)        
        # with torch.profiler.record_function("set_param"):
        atb_op.set_param(json.dumps(params.__dict__))
        atb_op_manager["SelfAttentionOperation"] = atb_op
        
    # with torch.profiler.record_function("set_op_name"):
    #     atb_op.set_op_name("SelfAttentionOperation")
    if attn_mask is not None:
        # import pdb; pdb.set_trace()
        if attn_mask.dtype != query.dtype:
            attn_mask = attn_mask.to(query.dtype)
        if q_seq_len.dtype != torch.int32:
            q_seq_len = q_seq_len.to(torch.int32)
        query = query.view(-1, num_q_heads * query.shape[-1])
        key = key.view(-1, num_kv_heads * key.shape[-1])
        value = value.view(-1, num_kv_heads * value.shape[-1])
        # mask要求f16/bf16, seq_len要求int32
        # with torch.profiler.record_function("execute_out_with_param"):
        # import pdb; pdb.set_trace()
        atb_op_manager["SelfAttentionOperation"].execute_out_with_param([query, key, value, attn_mask, q_seq_len.to(query.device)],[attn_output], json.dumps({"seqLen": seq_len_list, "hasMask": True,}))
    else:
        import pdb; pdb.set_trace()
        params = PagedAttentionPrefillParams(
        headNum=num_q_heads,
        kvHeadNum=num_kv_heads,
        qkScale=1.0 / math.sqrt(query.shape[-1]),
        qScale=1.0, 
        calcType=3, #paged encode
        maskType=0, #norm mask
        )    
        atb_op.set_param(json.dumps(params.__dict__))
        atb_op.execute_out_with_param([query, key, value],[attn_output], json.dumps({ "seqLen": seq_len_list, "hasMask": False,}))
    return attn_output



@register_ops(vendor_ops_registry)
def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    # import pdb; pdb.set_trace()
    if kv_indices.dtype != torch.int32:
        kv_indices = kv_indices.to(torch.int32)
    _, num_heads, head_dim = key.shape
    num_blocks, block_size, hidden_size = key_cache.shape
    key_cache_reshaped = key_cache.view(num_blocks, block_size, num_heads, head_dim)
    value_cache_reshaped = value_cache.view(num_blocks, block_size, num_heads, head_dim)
    fill_kv_cache_atb(key, value, key_cache_reshaped, value_cache_reshaped, kv_indices)
    # key_cache = key_cache.view(num_blocks, block_size, hidden_size)
    # value_cache = value_cache.view(num_blocks, block_size, hidden_size)
    return key_cache, value_cache

def fill_kv_cache_atb(key, value, key_cache, value_cache, kv_indices):
    kv_indices = kv_indices.flatten()
    if "ReshapeAndCacheOperation" not in atb_op_manager:
        atb_op = torch.classes.OperationTorch.OperationTorch("ReshapeAndCacheOperation")
        # with record_function("set_param"):
        atb_op.set_param(json.dumps({}))
        atb_op_manager["ReshapeAndCacheOperation"] = atb_op
    # with record_function("set_op_name"):
    #     atb_op.set_op_name("ReshapeAndCacheOperation")

    # with record_function("execute_out"):
    atb_op_manager["ReshapeAndCacheOperation"].execute_out([key, value, key_cache, value_cache, kv_indices], [key_cache, value_cache])
    # import pdb; pdb.set_trace()


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
    paged_decode_attentio_atb(query, key_cache, value_cache, block_table, block_size, kv_seq_len, num_q_heads, num_kv_heads, attn_output)
    return attn_output

def paged_decode_attentio_atb(query, key_cache, value_cache, block_table, block_size, kv_seq_len, num_q_heads, num_kv_heads, attn_output):
    # import pdb; pdb.set_trace()
    _, _, head_dim = query.shape
    num_blocks, _, hidden_size = key_cache.shape
    # import pdb; pdb.set_trace()
    key_cache_reshaped = key_cache.view(num_blocks, block_size, num_kv_heads, head_dim)
    value_cache_reshaped = value_cache.view(num_blocks, block_size, num_kv_heads, head_dim)
    contextLens = kv_seq_len.tolist()
    if "PagedAttentionOperation" not in atb_op_manager:
        atb_op = torch.classes.OperationTorch.OperationTorch("PagedAttentionOperation")
        params = PagedAttentionParams(headNum=num_q_heads, qkScale=1.0 / math.sqrt(head_dim), kvHeadNum=num_kv_heads)
        # with torch.profiler.record_function("set_param"):
        atb_op.set_param(json.dumps(params.__dict__))
        atb_op_manager["PagedAttentionOperation"] = atb_op
    # with torch.profiler.record_function("set_op_name"):
    #     atb_op.set_op_name("PagedAttentionOperation")

    # with torch.profiler.record_function("execute_out_with_param"):
    atb_op_manager["PagedAttentionOperation"].execute_out_with_param([query, key_cache_reshaped, value_cache_reshaped, block_table, kv_seq_len.to(torch.int32).to(query.device)], [attn_output], json.dumps({"contextLen":contextLens}))
    # import pdb; pdb.set_trace()
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
    # import pdb; pdb.set_trace()
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
    if "PagedPrefillAttentionOperation" not in atb_op_manager:
        atb_op = torch.classes.OperationTorch.OperationTorch("PagedAttentionOperation")
        params = PagedAttentionParams(headNum=num_q_heads, qkScale=1.0 / math.sqrt(query.shape[-1]), kvHeadNum=num_kv_heads, maskType=1)
        # with torch.profiler.record_function("set_param"):
        atb_op.set_param(json.dumps(params.__dict__))
        atb_op_manager["PagedPrefillAttentionOperation"] = atb_op
    
    if attn_output is None:
        attn_output = torch.empty_like(query).cuda()
    
    if attn_mask.dtype != query.dtype:
        attn_mask = attn_mask.to(query.dtype)
    atb_out_origin = torch.empty_like(query).cuda()
    block_table_atb = torch.cat([block_table] * q_seq_len, dim = 0).cuda()
    kv_seq_len_atb = torch.cat([kv_seq_len] * q_seq_len, dim = 0).cuda()
    import pdb; pdb.set_trace()
    atb_out = paged_prefill_attention_atb(query, key_cache, value_cache, block_table_atb, block_size, q_start_loc, q_seq_len, kv_seq_len_atb, num_q_heads, num_kv_heads, attn_mask[0], None, None, atb_out_origin)
    # # cann incre_fa don't support paged_attn when q_seq_len > 1
    # batch = q_start_loc.shape[0]
    # q_seq_len_list = q_seq_len.tolist()
    # kv_seq_len_list = kv_seq_len.tolist()
    # scale_value = 1.0 / math.sqrt(query.shape[-1])
    # query = query.contiguous()
    # for i in range(batch):
    #     start = q_start_loc[i]
    #     mask = attn_mask[i]
    #     for j in range(q_seq_len_list[i]):
    #         single_q = query[start + j : start + j + 1].view(1, 1, -1)
    #         single_o = attn_output[start + j : start + j + 1].view(1, 1, -1)
    #         torch.ops.npu_ext.npu_incre_flash_attention_v4_out(
    #             single_q,
    #             key_cache,
    #             value_cache,
    #             single_o,
    #             padding_mask=None,
    #             atten_mask=mask[j : j + 1],
    #             actual_seq_lengths=kv_seq_len_list[i : i + 1],
    #             antiquant_scale=None,
    #             antiquant_offset=None,
    #             block_table=block_table,
    #             dequant_scale1=None,
    #             quant_scale1=None,
    #             dequant_scale2=None,
    #             quant_scale2=None,
    #             quant_offset2=None,
    #             num_heads=num_q_heads,
    #             scale_value=scale_value,
    #             input_layout="BSH",
    #             num_key_value_heads=num_kv_heads,
    #             block_size=block_size,
    #             inner_precise=1,
    #         )
    # import pdb; pdb.set_trace()
    return atb_out


def paged_prefill_attention_atb(    
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
    attn_output: Optional[Tensor],):
    # import pdb; pdb.set_trace()
    _, _, head_dim = query.shape
    num_blocks, _, hidden_size = key_cache.shape
    key_cache_reshaped = key_cache.view(num_blocks, block_size, num_kv_heads, head_dim)
    value_cache_reshaped = value_cache.view(num_blocks, block_size, num_kv_heads, head_dim)
    contextLens = kv_seq_len.tolist()
    # import pdb; pdb.set_trace()
    atb_op_manager["PagedPrefillAttentionOperation"].execute_out_with_param([query, key_cache_reshaped, value_cache_reshaped, block_table, kv_seq_len.to(torch.int32).to(query.device), attn_mask], [attn_output], json.dumps({"contextLen":contextLens}))
    return attn_output

@dataclass
class RmsNormParams:
    epsilon: float

@register_ops(vendor_ops_registry)
def rms_norm(hidden_states: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    if "RmsNormOperation" not in atb_op_manager:
        atb_op = torch.classes.OperationTorch.OperationTorch("RmsNormOperation")
        params = RmsNormParams(epsilon=epsilon)
        # with torch.profiler.record_function("set_param"):
        atb_op.set_param(json.dumps(params.__dict__))
        atb_op_manager["RmsNormOperation"] = atb_op

    # import pdb; pdb.set_trace()
    # with torch.profiler.record_function("execute"):
    out = atb_op_manager["RmsNormOperation"].execute([hidden_states, weight])[0]
    return out


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