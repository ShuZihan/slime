from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
megatron = pytest.importorskip("megatron")

import torch.nn as nn
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from slime_plugins.models.qwen4_exp.model import Qwen4ExpTransformerLayer
from slime_plugins.models.qwen4_exp.reference import Qwen4ExpPackedLayout

from test_qwen4_exp_reference import tiny_config


class FakeMlp(nn.Module):
    def __init__(self, config, pg_collection=None):
        super().__init__()
        del pg_collection
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def set_layer_number(self, layer_number):
        self.layer_number = layer_number

    def forward(self, hidden_states, padding_mask=None):
        del padding_mask
        return self.proj(hidden_states), None


@pytest.mark.unit
def test_megatron_layer_expands_mhc_streams_and_preserves_gradient_path():
    p0_config = tiny_config()
    mcore_config = TransformerConfig(
        num_layers=4,
        hidden_size=p0_config.hidden_size,
        num_attention_heads=p0_config.num_attention_heads,
        num_query_groups=p0_config.num_key_value_heads,
        ffn_hidden_size=16,
        kv_channels=p0_config.head_dim,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        transformer_impl="local",
    )
    layer = Qwen4ExpTransformerLayer(
        config=mcore_config,
        layer_number=4,
        args=SimpleNamespace(),
        hf_config=SimpleNamespace(text_config=SimpleNamespace()),
        p0_config=p0_config,
        layer_type="qwen_sparse_attention",
        mlp_spec=ModuleSpec(module=FakeMlp),
        ple_layer_index=None,
        pg_collection=SimpleNamespace(tp=None),
    )
    layout = Qwen4ExpPackedLayout.from_cu_seqlens(6, torch.tensor([0, 3, 6]))
    packed_seq_params = SimpleNamespace(
        qwen4_exp_layout=layout,
        qwen4_exp_input_ids=torch.arange(6),
    )
    hidden_states = torch.randn(6, 1, p0_config.hidden_size, requires_grad=True)

    output, context = layer(hidden_states, packed_seq_params=packed_seq_params)
    output.square().mean().backward()

    assert output.shape == (6, 1, p0_config.hyper_connection_width)
    assert context is None
    assert hidden_states.grad is not None
    assert layer.mlp.proj.weight.grad is not None
    assert all(parameter.grad is None for parameter in layer.self_attention.indexer.parameters())
