from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from slime.backends.megatron_utils.hf_to_megatron.qwen4_exp import Qwen4ExpHfLoader, _merge_gated_qkv
from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
from slime.backends.megatron_utils.megatron_to_hf.qwen4_exp import (
    _unpack_gated_qkv,
    convert_qwen4_exp_to_hf,
)

from test_qwen4_exp_reference import tiny_config


class FakeReader:
    def __init__(self, tensors):
        self.tensors = tensors
        self.weight_map = {name: f"fake-{idx}.safetensors" for idx, name in enumerate(tensors)}
        self.loaded = []
        self.sliced = []

    def get_shape(self, name):
        return tuple(self.tensors[name].shape)

    def get_tensor(self, name):
        self.loaded.append(name)
        return self.tensors[name]

    def get_tensor_slice(self, name, index):
        self.sliced.append((name, index))
        return self.tensors[name][index]


@pytest.mark.unit
def test_safetensor_reader_materializes_only_the_requested_rows(tmp_path):
    import json

    from safetensors.torch import save_file

    tensor = torch.arange(48, dtype=torch.float32).reshape(12, 4)
    save_file({"weight": tensor}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model.safetensors"}}),
        encoding="utf-8",
    )

    reader = SafetensorReader(tmp_path)

    assert reader.get_shape("weight") == (12, 4)
    torch.testing.assert_close(reader.get_tensor_slice("weight", slice(3, 7)), tensor[3:7])


@pytest.mark.unit
def test_gated_qkv_mapping_round_trips_checkpoint_head_order():
    args = SimpleNamespace(
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=4,
        hidden_size=8,
    )
    q_gate = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
    key = 1000 + torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    value = 2000 + torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    reader = FakeReader({"layer.q_proj.weight": q_gate, "layer.k_proj.weight": key, "layer.v_proj.weight": value})
    config = SimpleNamespace(num_attention_heads=2, num_key_value_heads=1, head_dim=4)

    packed = _merge_gated_qkv(reader, "layer", config)
    actual_q_gate, actual_key, actual_value = _unpack_gated_qkv(args, packed)

    torch.testing.assert_close(actual_q_gate, q_gate)
    torch.testing.assert_close(actual_key, key)
    torch.testing.assert_close(actual_value, value)


@pytest.mark.unit
def test_ple_loader_materializes_only_overlapping_source_shards():
    config = tiny_config()
    hf_config = {"text_config": dict(config.__dict__)}
    # Translate the compact test dataclass back to public config field names.
    hf_config["text_config"]["hc_count"] = hf_config["text_config"].pop("hyper_connection_count")
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    from slime_plugins.models.qwen4_exp.reference import Qwen4ExpNGramLayout

    layout = Qwen4ExpNGramLayout.from_config(config)
    rows_per_source_shard = (layout.padded_vocab_size + config.split_ngram_parts - 1) // config.split_ngram_parts
    table = torch.arange(layout.padded_vocab_size * 2, dtype=torch.float32).reshape(layout.padded_vocab_size, 2)
    tensors = {}
    for shard_idx in range(config.split_ngram_parts):
        start = shard_idx * rows_per_source_shard
        end = min(start + rows_per_source_shard, layout.padded_vocab_size)
        tensors[f"{prefix}.shard_{shard_idx}.weight"] = table[start:end]
    reader = FakeReader(tensors)

    parameter = torch.nn.Parameter(torch.empty(rows_per_source_shard, 2), requires_grad=False)
    parameter.qwen4_vocab_start_index = rows_per_source_shard
    parameter.qwen4_vocab_end_index = 2 * rows_per_source_shard
    loader = Qwen4ExpHfLoader()

    handled = loader.load_parameter(
        SimpleNamespace(),
        "module.module.decoder.layers.1.ple.ple_embedding.ngram_embedding.weight",
        parameter,
        reader,
        hf_config,
    )

    assert handled
    torch.testing.assert_close(parameter, table[rows_per_source_shard : 2 * rows_per_source_shard])
    assert reader.loaded == []
    assert len(reader.sliced) == 1
    source_name, source_slice = reader.sliced[0]
    assert source_name == f"{prefix}.shard_1.weight"
    assert source_slice.start == 0
    assert source_slice.stop == rows_per_source_shard


@pytest.mark.unit
def test_ple_config_derived_buffers_match_checkpoint_exactly():
    from slime_plugins.models.qwen4_exp.reference import Qwen4ExpPLE

    config = tiny_config()
    hf_config = {"text_config": dict(config.__dict__)}
    hf_config["text_config"]["hc_count"] = hf_config["text_config"].pop("hyper_connection_count")

    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Module()])
    model.decoder.layers[1].ple = torch.nn.Module()
    embedding = Qwen4ExpPLE(config).ple_embedding
    model.decoder.layers[1].ple.ple_embedding = embedding

    prefix = "model.language_model.layers.1.ple.ple_embedding"
    tensors = {
        f"{prefix}.layer_multipliers": embedding.layer_multipliers.clone(),
        f"{prefix}.ngram_heads_vocab_sizes": embedding.ngram_heads_vocab_sizes.clone(),
        f"{prefix}.ngram_heads_offsets": embedding.ngram_heads_offsets.clone(),
    }
    reader = FakeReader(tensors)

    Qwen4ExpHfLoader().finalize_load(SimpleNamespace(), [model], reader, hf_config)

    assert set(reader.loaded) == set(tensors)


@pytest.mark.unit
def test_grouped_moe_loader_and_online_converter_use_global_expert_id():
    prefix = "model.language_model.layers.7"
    grouped_fc1 = torch.arange(8 * 12 * 5, dtype=torch.float32).reshape(8, 12, 5)
    grouped_fc2 = 1000 + torch.arange(8 * 5 * 6, dtype=torch.float32).reshape(8, 5, 6)
    reader = FakeReader(
        {
            f"{prefix}.mlp.experts.gate_up_proj": grouped_fc1,
            f"{prefix}.mlp.experts.down_proj": grouped_fc2,
        }
    )
    loader = Qwen4ExpHfLoader()

    actual_fc1 = loader(
        "module.module.decoder.layers.7.mlp.experts.linear_fc1.weight6",
        reader,
        SimpleNamespace(),
    )
    actual_fc2 = loader(
        "module.module.decoder.layers.7.mlp.experts.linear_fc2.weight6",
        reader,
        SimpleNamespace(),
    )
    torch.testing.assert_close(actual_fc1, grouped_fc1[6])
    torch.testing.assert_close(actual_fc2, grouped_fc2[6])

    args = SimpleNamespace()
    converted_fc1 = convert_qwen4_exp_to_hf(
        args,
        "module.module.decoder.layers.7.mlp.experts.linear_fc1.weight6",
        actual_fc1,
    )
    converted_fc2 = convert_qwen4_exp_to_hf(
        args,
        "module.module.decoder.layers.7.mlp.experts.linear_fc2.weight6",
        actual_fc2,
    )
    assert [name for name, _ in converted_fc1] == [
        f"{prefix}.mlp.experts.6.gate_proj.weight",
        f"{prefix}.mlp.experts.6.up_proj.weight",
    ]
    assert [name for name, _ in converted_fc2] == [f"{prefix}.mlp.experts.6.down_proj.weight"]
    torch.testing.assert_close(torch.cat([tensor for _, tensor in converted_fc1], dim=0), grouped_fc1[6])
    torch.testing.assert_close(converted_fc2[0][1], grouped_fc2[6])
    assert reader.loaded == []
    assert reader.sliced == [
        (f"{prefix}.mlp.experts.gate_up_proj", 6),
        (f"{prefix}.mlp.experts.down_proj", 6),
    ]


@pytest.mark.unit
def test_direct_trainable_and_static_mappings_cover_every_qwen4_exp_subtree():
    prefix = "model.language_model.layers.3"
    tensors = {
        "model.language_model.embed_tokens.weight": torch.randn(7, 8),
        "lm_head.weight": torch.randn(7, 8),
        "model.language_model.hyper_connection_mixer.hc_norm.weight": torch.randn(16),
        f"{prefix}.attn_hyper_connection.input_mix_weight_down.weight": torch.randn(4, 16),
        f"{prefix}.linear_attn.in_proj_qkv.weight": torch.randn(16, 8),
        f"{prefix}.self_attn.q_proj.weight": torch.randn(16, 8),
        f"{prefix}.self_attn.indexer.index_qk_proj.weight": torch.randn(12, 8),
        f"{prefix}.mlp.gate.weight": torch.randn(4, 8),
        f"{prefix}.mlp.shared_expert.gate_proj.weight": torch.randn(8, 8),
        f"{prefix}.mlp.shared_expert.up_proj.weight": torch.randn(8, 8),
        f"{prefix}.mlp.shared_expert.down_proj.weight": torch.randn(8, 8),
        f"{prefix}.mlp.shared_expert_gate.weight": torch.randn(1, 8),
    }
    reader = FakeReader(tensors)
    loader = Qwen4ExpHfLoader()
    cases = {
        "module.module.embedding.word_embeddings.weight": ["model.language_model.embed_tokens.weight"],
        "module.module.output_layer.weight": ["lm_head.weight"],
        "module.module.decoder.final_layernorm.hc_norm.weight": [
            "model.language_model.hyper_connection_mixer.hc_norm.weight"
        ],
        "module.module.decoder.layers.3.attn_hyper_connection.input_mix_weight_down.weight": [
            f"{prefix}.attn_hyper_connection.input_mix_weight_down.weight"
        ],
        "module.module.decoder.layers.3.linear_attn.in_proj_qkv.weight": [
            f"{prefix}.linear_attn.in_proj_qkv.weight"
        ],
        "module.module.decoder.layers.3.self_attention.q_proj.weight": [f"{prefix}.self_attn.q_proj.weight"],
        "module.module.decoder.layers.3.self_attention.indexer.index_qk_proj.weight": [
            f"{prefix}.self_attn.indexer.index_qk_proj.weight"
        ],
        "module.module.decoder.layers.3.mlp.router.weight": [f"{prefix}.mlp.gate.weight"],
        "module.module.decoder.layers.3.mlp.shared_experts.linear_fc1.weight": [
            f"{prefix}.mlp.shared_expert.gate_proj.weight",
            f"{prefix}.mlp.shared_expert.up_proj.weight",
        ],
        "module.module.decoder.layers.3.mlp.shared_experts.linear_fc2.weight": [
            f"{prefix}.mlp.shared_expert.down_proj.weight"
        ],
        "module.module.decoder.layers.3.mlp.shared_experts.gate_weight": [
            f"{prefix}.mlp.shared_expert_gate.weight"
        ],
    }
    args = SimpleNamespace(
        kv_channels=4,
        hidden_size=8,
        num_attention_heads=2,
        num_query_groups=1,
    )

    for megatron_name, source_names in cases.items():
        loaded = loader(megatron_name, reader, SimpleNamespace())
        expected = torch.cat([tensors[name] for name in source_names], dim=0)
        torch.testing.assert_close(loaded, expected)
        if ".indexer." in megatron_name:
            continue
        converted = convert_qwen4_exp_to_hf(args, megatron_name, loaded)
        assert [name for name, _ in converted] == source_names
        for (_, actual), source_name in zip(converted, source_names, strict=True):
            torch.testing.assert_close(actual, tensors[source_name])
