"""HF file round trips with scripted transport and the real tensor mappings."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from safetensors.torch import save_file
from test_qwen4_exp_reference import tiny_config

from slime.backends.megatron_utils.hf_checkpoint_saver import save_hf_model_to_path
from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
from slime.backends.megatron_utils.hf_to_megatron.qwen4_exp import Qwen4ExpHfLoader
from slime.backends.megatron_utils.megatron_to_hf.qwen4_exp import convert_qwen4_exp_to_hf
from slime_plugins.models.qwen4_exp.reference import Qwen4ExpNGramLayout, Qwen4ExpPLE


def _source_checkpoint(path):
    path.mkdir()
    config = tiny_config()
    hf_text = dict(config.__dict__)
    hf_text["hc_count"] = hf_text.pop("hyper_connection_count")
    (path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": hf_text}))
    prefix = "model.language_model.layers.1"
    tensors = {
        "lm_head.weight": torch.ones(64, 8),
        f"{prefix}.mlp.experts.gate_up_proj": torch.zeros(4, 16, 8),
        f"{prefix}.mlp.experts.down_proj": torch.zeros(4, 8, 8),
        "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight": torch.arange(4).float(),
        "model.visual.patch_embed.weight": torch.full((2, 3), 9.0),
        "mtp.fc.weight": torch.full((2, 3), 11.0),
    }
    ple = Qwen4ExpPLE(config).ple_embedding
    for name in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
        tensors[f"{prefix}.ple.ple_embedding.{name}"] = getattr(ple, name)
    rows = Qwen4ExpNGramLayout.from_config(config).padded_vocab_size
    table = torch.arange(rows * 2).float().reshape(rows, 2)
    for index, shard in enumerate(table.chunk(config.split_ngram_parts)):
        tensors[f"{prefix}.ple.ple_embedding.ngram_embedding.shard_{index}.weight"] = shard.contiguous()
    save_file(tensors, path / "model.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    return tensors, table, {"text_config": hf_text}


def _live_weights():
    weights = [("output_layer.weight", torch.full((64, 8), 3.0))]
    for expert in (2, 0, 3, 1):  # Global IDs, deliberately out of arrival order.
        for projection, shape in ((1, (16, 8)), (2, (8, 8))):
            weights.append(
                (
                    f"decoder.layers.1.mlp.experts.linear_fc{projection}.weight{expert}",
                    torch.arange(shape[0] * shape[1]).reshape(shape).float() + 1000 * expert + 100 * projection,
                )
            )
    return weights


def _script_transport(monkeypatch, weights):
    class ScriptedIterator:
        def __init__(self, **kwargs):
            # The production iterator applies this before any TP/EP collection.
            assert kwargs.get("include_static") is False, "HF export must not gather the frozen PLE table"

        def get_hf_weight_chunks(self, local_weights, *, should_convert_chunk, **kwargs):
            assert dict(weights).keys() == local_weights.keys()
            for index, (name, tensor) in enumerate(weights):
                assert should_convert_chunk(index)
                # Split gate/up across transport chunks to exercise buffering.
                for converted in convert_qwen4_exp_to_hf(SimpleNamespace(), name, tensor):
                    yield [converted]

    package = "slime.backends.megatron_utils.update_weight"
    monkeypatch.setitem(
        sys.modules, f"{package}.hf_weight_iterator_direct", SimpleNamespace(HfWeightIteratorDirect=ScriptedIterator)
    )
    monkeypatch.setitem(sys.modules, f"{package}.common", SimpleNamespace(named_params_and_buffers=lambda *_: weights))


@pytest.mark.unit
def test_save_hf_restores_source_schema_and_reloads_live_experts_and_static_ple(tmp_path, monkeypatch):
    source, table, hf_config = _source_checkpoint(tmp_path / "source")
    weights = _live_weights()
    _script_transport(monkeypatch, weights)
    output = tmp_path / "export"
    save_hf_model_to_path(SimpleNamespace(hf_checkpoint=str(tmp_path / "source")), output, [], model_name="qwen4_exp")
    reader = SafetensorReader(output)
    assert set(reader.weight_map) == set(source)
    loader = Qwen4ExpHfLoader()
    for name, expected in weights:
        torch.testing.assert_close(loader(name, reader, SimpleNamespace()), expected, rtol=0, atol=0)
    for name, expected in source.items():
        if ".mlp.experts." not in name and name != "lm_head.weight":
            torch.testing.assert_close(reader.get_tensor(name), expected, rtol=0, atol=0)
    shard = torch.nn.Parameter(torch.empty_like(table[7:33]), requires_grad=False)
    shard.qwen4_vocab_start_index, shard.qwen4_vocab_end_index = 7, 33
    assert loader.load_parameter(
        None, "decoder.layers.1.ple.ple_embedding.ngram_embedding.weight", shard, reader, hf_config
    )
    torch.testing.assert_close(shard, table[7:33], rtol=0, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("fault", ["missing", "duplicate", "shape"])
def test_export_rejects_incomplete_or_corrupt_live_weights(tmp_path, monkeypatch, fault):
    _source_checkpoint(tmp_path / "source")
    weights = _live_weights()
    if fault == "missing":
        weights.pop()
    elif fault == "duplicate":
        weights.append(weights[1])
    else:
        weights[1] = (weights[1][0], torch.zeros(14, 8))
    _script_transport(monkeypatch, weights)
    output = tmp_path / "export"
    with pytest.raises(RuntimeError, match="(missing|Duplicate|shape)"):
        save_hf_model_to_path(
            SimpleNamespace(hf_checkpoint=str(tmp_path / "source")), output, [], model_name="qwen4_exp"
        )
    assert not (output / "model.safetensors.index.json").exists()
