from __future__ import annotations

import re

import torch


def _unpack_gated_qkv(args, parameter: torch.Tensor):
    head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    queries_per_group = args.num_attention_heads // args.num_query_groups
    parameter = parameter.view(args.num_query_groups, 2 * queries_per_group + 2, head_dim, args.hidden_size)
    query_gate, key, value = torch.split(parameter, [2 * queries_per_group, 1, 1], dim=1)
    query_gate = (
        query_gate.reshape(args.num_query_groups, 2, queries_per_group, head_dim, args.hidden_size)
        .transpose(1, 2)
        .reshape(-1, args.hidden_size)
    )
    return query_gate, key.reshape(-1, args.hidden_size), value.reshape(-1, args.hidden_size)


def convert_qwen4_exp_to_hf(args, name, parameter):
    """Map one full Megatron tensor to checkpoint-shaped Qwen4-Exp tensors."""

    while name.startswith("module."):
        name = name.removeprefix("module.")
    name = name.removeprefix("language_model.")

    root_mapping = {
        "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "output_layer.weight": "lm_head.weight",
    }
    if name in root_mapping:
        return [(root_mapping[name], parameter)]
    if name.startswith("decoder.final_layernorm."):
        suffix = name.removeprefix("decoder.final_layernorm.")
        return [(f"model.language_model.hyper_connection_mixer.{suffix}", parameter)]
    if name.startswith("hyper_connection_mixer."):
        return [(f"model.language_model.{name}", parameter)]

    layer_match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not layer_match:
        raise ValueError(f"unknown Qwen4-Exp parameter name: {name}")
    layer_idx, rest = layer_match.groups()
    prefix = f"model.language_model.layers.{layer_idx}"

    for direct_prefix in ("attn_hyper_connection.", "mlp_hyper_connection.", "linear_attn.", "ple."):
        if rest.startswith(direct_prefix):
            if rest == "ple.ple_embedding.ngram_embedding.weight":
                raise ValueError(
                    "the frozen PLE table uses source checkpoint shards and is excluded from online updates"
                )
            return [(f"{prefix}.{rest}", parameter)]

    attention_mapping = {
        "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
        "self_attention.q_proj.weight": "self_attn.q_proj.weight",
        "self_attention.k_proj.weight": "self_attn.k_proj.weight",
        "self_attention.v_proj.weight": "self_attn.v_proj.weight",
        "self_attention.o_proj.weight": "self_attn.o_proj.weight",
        "self_attention.q_norm.weight": "self_attn.q_norm.weight",
        "self_attention.k_norm.weight": "self_attn.k_norm.weight",
        "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
        "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
        "self_attention.indexer.index_qk_proj.weight": "self_attn.indexer.index_qk_proj.weight",
        "self_attention.indexer.q_layernorm.weight": "self_attn.indexer.q_layernorm.weight",
        "self_attention.indexer.k_layernorm.weight": "self_attn.indexer.k_layernorm.weight",
    }
    if rest == "self_attention.linear_qkv.weight":
        query_gate, key, value = _unpack_gated_qkv(args, parameter)
        return [
            (f"{prefix}.self_attn.q_proj.weight", query_gate),
            (f"{prefix}.self_attn.k_proj.weight", key),
            (f"{prefix}.self_attn.v_proj.weight", value),
        ]
    if rest in attention_mapping:
        return [(f"{prefix}.{attention_mapping[rest]}", parameter)]

    if rest == "mlp.router.weight":
        return [(f"{prefix}.mlp.gate.weight", parameter)]
    if rest == "mlp.experts.linear_fc1":
        return [(f"{prefix}.mlp.experts.gate_up_proj", parameter)]
    if rest == "mlp.experts.linear_fc2":
        return [(f"{prefix}.mlp.experts.down_proj", parameter)]

    expert_match = re.fullmatch(r"mlp\.experts\.(linear_fc[12])\.weight(\d+)", rest)
    if expert_match:
        projection, expert_idx = expert_match.groups()
        if projection == "linear_fc1":
            gate, up = parameter.chunk(2, dim=0)
            return [
                (f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight", gate),
                (f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight", up),
            ]
        return [(f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight", parameter)]

    local_expert_match = re.fullmatch(r"mlp\.experts\.local_experts\.(\d+)\.(linear_fc[12])\.weight", rest)
    if local_expert_match:
        expert_idx, projection = local_expert_match.groups()
        if projection == "linear_fc1":
            gate, up = parameter.chunk(2, dim=0)
            return [
                (f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight", gate),
                (f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight", up),
            ]
        return [(f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight", parameter)]

    shared_match = re.fullmatch(r"mlp\.shared_experts\.(.+)", rest)
    if shared_match:
        shared_name = shared_match.group(1)
        if shared_name == "linear_fc1.weight":
            gate, up = parameter.chunk(2, dim=0)
            return [
                (f"{prefix}.mlp.shared_expert.gate_proj.weight", gate),
                (f"{prefix}.mlp.shared_expert.up_proj.weight", up),
            ]
        if shared_name == "linear_fc2.weight":
            return [(f"{prefix}.mlp.shared_expert.down_proj.weight", parameter)]
        if shared_name == "gate_weight":
            return [(f"{prefix}.mlp.shared_expert_gate.weight", parameter)]
    raise ValueError(f"unknown Qwen4-Exp parameter name: {name}")
