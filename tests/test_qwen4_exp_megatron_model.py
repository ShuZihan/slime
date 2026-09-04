from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
megatron = pytest.importorskip("megatron")

import torch.distributed as dist
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig

import slime_plugins.models.qwen4_exp.model as qwen4_model
from slime.backends.megatron_utils.megatron_to_hf.qwen4_exp import convert_qwen4_exp_to_hf
from slime_plugins.models.qwen4_exp.lifecycle import should_online_update_megatron_parameter

from test_qwen4_exp_reference import tiny_config


class FakeGatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx, args=None):
        super().__init__()
        del layer_idx, args
        self.in_proj_qkv = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, cu_seqlens=None):
        del cu_seqlens
        return self.in_proj_qkv(hidden_states)


class CpuMlp(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, padding_mask=None):
        del padding_mask
        return self.proj(hidden_states), None


def _mcore_config(p0_config):
    return TransformerConfig(
        num_layers=p0_config.num_hidden_layers,
        hidden_size=p0_config.hidden_size,
        num_attention_heads=p0_config.num_attention_heads,
        num_query_groups=p0_config.num_key_value_heads,
        ffn_hidden_size=16,
        kv_channels=p0_config.head_dim,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        transformer_impl="local",
        normalization="RMSNorm",
        layernorm_epsilon=p0_config.rms_norm_eps,
        gated_linear_unit=True,
        num_moe_experts=p0_config.num_experts,
        moe_ffn_hidden_size=p0_config.moe_intermediate_size,
        moe_router_topk=p0_config.num_experts_per_tok,
        moe_shared_expert_intermediate_size=p0_config.shared_expert_intermediate_size,
        moe_shared_expert_gate=True,
        moe_token_dispatcher_type="allgather",
        hidden_dropout=0,
        attention_dropout=0,
        add_bias_linear=False,
    )


def _runtime_args(p0_config):
    return SimpleNamespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        mtp_num_layers=0,
        enable_mtp_training=False,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        num_experts=p0_config.num_experts,
        moe_router_topk=p0_config.num_experts_per_tok,
        moe_router_pre_softmax=False,
        untie_embeddings_and_output_weights=True,
        hf_checkpoint="unused",
        transformer_impl="local",
        seq_length=4,
        padded_vocab_size=p0_config.vocab_size,
        max_position_embeddings=32,
        fp16_lm_cross_entropy=False,
    )


@pytest.mark.unit
def test_runtime_contract_rejects_router_probability_mismatch():
    p0_config = tiny_config()
    args = _runtime_args(p0_config)
    args.moe_router_pre_softmax = True

    with pytest.raises(ValueError, match="router normalization"):
        qwen4_model._validate_p0_runtime(args, p0_config)


@pytest.mark.unit
def test_runtime_contract_rejects_sequence_above_qsa_budget():
    p0_config = tiny_config()
    args = _runtime_args(p0_config)
    args.seq_length = p0_config.indexer_budget + 1

    with pytest.raises(ValueError, match="QSA budget"):
        qwen4_model._validate_p0_runtime(args, p0_config)


@pytest.mark.integration
def test_full_megatron_shell_builds_and_runs_tiny_packed_graph(monkeypatch):
    if dist.is_initialized():
        pytest.skip("test owns its world-size-one process group")
    p0_config = tiny_config()
    hf_text_config = SimpleNamespace(**p0_config.__dict__)
    hf_text_config.hc_count = p0_config.hyper_connection_count
    monkeypatch.setattr(qwen4_model, "Qwen3_5GatedDeltaNet", FakeGatedDeltaNet)
    monkeypatch.setattr(
        qwen4_model.Qwen4ExpP0Config,
        "validate_public_release_contract",
        lambda _config: None,
    )
    monkeypatch.setattr(
        qwen4_model,
        "_load_hf_config",
        lambda _path: SimpleNamespace(text_config=hf_text_config),
    )

    file_descriptor, rendezvous_path = tempfile.mkstemp(prefix="qwen4-exp-gloo-")
    os.close(file_descriptor)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous_path}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    try:
        model = qwen4_model.get_qwen4_exp_model_provider(
            _runtime_args(p0_config), _mcore_config(p0_config), None
        )()
        converted_names = []
        for name, parameter in model.named_parameters():
            if should_online_update_megatron_parameter("qwen4_exp", name):
                converted_names.extend(
                    hf_name for hf_name, _tensor in convert_qwen4_exp_to_hf(_runtime_args(p0_config), name, parameter)
                )
        assert "model.language_model.embed_tokens.weight" in converted_names
        assert "lm_head.weight" in converted_names
        assert "model.language_model.hyper_connection_mixer.hc_norm.weight" in converted_names
        assert not any(".indexer." in name for name in converted_names)

        # MCore's MoE router places a CPU-initialized weight on CUDA during its
        # first forward.  The model-construction path above still builds the real
        # MoE; this replacement keeps the local CPU graph executable.
        for layer in model.decoder.layers:
            layer.mlp = CpuMlp(p0_config.hidden_size)

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=3,
            max_seqlen_kv=3,
            qkv_format="thd",
        )

        logits = model(input_ids, None, None, packed_seq_params=packed_seq_params)
        logits.float().square().mean().backward()

        assert logits.shape == (1, 6, p0_config.vocab_size)
        assert isinstance(model.decoder.final_layernorm, qwen4_model.Qwen4ExpGatedResidual)
        table_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if "ple_embedding.ngram_embedding.weight" in name
        ]
        assert len(table_parameters) == 1
        assert not table_parameters[0][1].requires_grad
        assert hasattr(packed_seq_params, "qwen4_exp_layout")
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
