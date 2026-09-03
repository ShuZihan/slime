from __future__ import annotations

import pytest

from slime_plugins.models.qwen4_exp.config import Qwen4ExpP0Config
from slime_plugins.models.qwen4_exp.lifecycle import (
    ParameterLifecycle,
    build_lifecycle_manifest,
    lifecycle_summary,
    online_update_source_names,
    should_online_update_megatron_parameter,
)


def public_text_config():
    return {
        "text_config": {
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "rms_norm_eps": 1e-6,
            "hc_count": 4,
            "hc_lowrank": 320,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_theta": 10_000_000},
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "ple_layer_ids": [2],
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 20_000_000,
            "make_ngram_vocab_size_divisible_by": 128,
            "split_ngram_parts": 128,
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "pad_token_id": None,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "norm_topk_prob": True,
            "hidden_act": "silu",
            "output_gate_type": "sigmoid",
            "attention_bias": False,
            "tie_word_embeddings": False,
        }
    }


@pytest.mark.unit
def test_config_reads_public_names_and_derives_p0_contract():
    raw = public_text_config()
    raw["text_config"]["layer_types"] *= 12

    config = Qwen4ExpP0Config.from_hf_config(raw)

    assert config.layer_types[3] == "qwen_sparse_attention"
    assert config.hyper_connection_width == 10240
    assert config.ple_num_heads == 16
    assert config.index_block_topk == 512
    assert config.rope_theta == 10_000_000
    assert config.pad_token_id == config.eos_token_id == 248044
    assert config.norm_topk_prob
    assert config.output_gate_type == "sigmoid"


@pytest.mark.unit
def test_public_null_seed_and_rope_theta_resolve_to_checkpoint_defaults():
    raw = public_text_config()
    raw["text_config"]["layer_types"] *= 12
    raw["text_config"]["seed"] = None
    raw["text_config"]["rope_theta"] = None

    config = Qwen4ExpP0Config.from_hf_config(raw)

    assert config.seed == 1234
    assert config.rope_theta == 10_000_000
    config.validate_public_release_contract()


@pytest.mark.unit
def test_config_rejects_ple_on_qsa_layer():
    raw = public_text_config()
    raw["text_config"]["num_hidden_layers"] = 4
    raw["text_config"]["ple_layer_ids"] = [4]

    with pytest.raises(ValueError, match="PLE is supported only"):
        Qwen4ExpP0Config.from_hf_config(raw)


@pytest.mark.unit
def test_lifecycle_manifest_covers_p0_checkpoint_and_sync_set():
    raw = public_text_config()
    raw["text_config"]["layer_types"] *= 12
    config = Qwen4ExpP0Config.from_hf_config(raw)
    weight_map = {
        "model.language_model.embed_tokens.weight": "model-00001.safetensors",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "model-00002.safetensors",
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers": "model-00002.safetensors",
        "model.visual.blocks.0.attn.qkv.weight": "model-00131.safetensors",
        "mtp.fc_embedding.weight": "model-00131.safetensors",
        "lm_head.weight": "model-00131.safetensors",
    }
    for shard_idx in range(config.split_ngram_parts):
        weight_map[
            f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard_idx}.weight"
        ] = f"model-{shard_idx + 3:05d}.safetensors"
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_type == "qwen_sparse_attention":
            for suffix in ("index_qk_proj.weight", "q_layernorm.weight", "k_layernorm.weight"):
                weight_map[f"model.language_model.layers.{layer_idx}.self_attn.indexer.{suffix}"] = (
                    "model-00100.safetensors"
                )

    records = build_lifecycle_manifest(weight_map, config)
    summary = lifecycle_summary(records)
    update_names = online_update_source_names(records)

    assert summary[ParameterLifecycle.STATIC_SHARED.value] == 128 + 12 * 3
    assert summary[ParameterLifecycle.DERIVED_BUFFER.value] == 1
    assert summary[ParameterLifecycle.DISABLED.value] == 2
    assert "lm_head.weight" in update_names
    assert not any("ngram_embedding.shard_" in name for name in update_names)
    assert not any(".indexer." in name for name in update_names)


@pytest.mark.unit
def test_online_update_filter_removes_static_megatron_parameters_only_for_qwen4_exp():
    table = "module.module.decoder.layers.1.ple.ple_embedding.ngram_embedding.weight"
    derived = "module.module.decoder.layers.1.ple.ple_embedding.layer_multipliers"
    indexer = "module.module.decoder.layers.3.self_attention.indexer.index_qk_proj.weight"
    trainable = "module.module.decoder.layers.3.self_attention.linear_qkv.weight"

    assert not should_online_update_megatron_parameter("Qwen4ExpForConditionalGeneration", table)
    assert not should_online_update_megatron_parameter("qwen4_exp", derived)
    assert not should_online_update_megatron_parameter("qwen4_exp", indexer)
    assert should_online_update_megatron_parameter("qwen4_exp", trainable)
    assert should_online_update_megatron_parameter("qwen3_5", table)
