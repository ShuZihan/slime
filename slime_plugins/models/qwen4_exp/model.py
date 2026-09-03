"""Megatron adapter for the Qwen4-Exp text-only P0 training graph."""

from __future__ import annotations

import copy
from typing import Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import BaseTransformerLayer

from slime.utils import accelerator

from ..hf_attention import _load_hf_config
from ..qwen3_5 import Qwen3_5GatedDeltaNet
from .config import Qwen4ExpP0Config
from .reference import (
    Qwen4ExpGatedResidual,
    Qwen4ExpPLE,
    Qwen4ExpPackedLayout,
    Qwen4ExpSparseAttentionReference,
    inject_residual,
)


def _text_config(config):
    return getattr(config, "text_config", config)


def _module_device(config):
    if config.use_cpu_initialization:
        return torch.device("cpu")
    return accelerator.current_device()


def _place_replicated_module(module: torch.nn.Module, config) -> torch.nn.Module:
    return module.to(device=_module_device(config), dtype=config.params_dtype)


def _gather_sequence_parallel(hidden_states: torch.Tensor, config, tp_group):
    if not config.sequence_parallel:
        return hidden_states
    return tensor_parallel.gather_from_sequence_parallel_region(
        hidden_states,
        tensor_parallel_output_grad=False,
        group=tp_group,
    )


def _scatter_sequence_parallel(hidden_states: torch.Tensor, config, tp_group):
    if not config.sequence_parallel:
        return hidden_states
    return tensor_parallel.scatter_to_sequence_parallel_region(hidden_states, group=tp_group)


class Qwen4ExpTransformerLayer(MegatronModule, BaseTransformerLayer):
    """One mHC decoder layer with native Megatron MoE and reference Qwen operators."""

    def __init__(
        self,
        config,
        layer_number: int,
        args,
        hf_config,
        p0_config: Qwen4ExpP0Config,
        layer_type: str,
        mlp_spec: ModuleSpec,
        ple_layer_index: Optional[int],
        pg_collection=None,
        vp_stage=None,
    ):
        MegatronModule.__init__(self, config=config)
        BaseTransformerLayer.__init__(self)
        del vp_stage
        self.layer_number = layer_number
        self.args = args
        self.p0_config = p0_config
        self.layer_type = layer_type
        self.tp_group = pg_collection.tp

        self.attn_hyper_connection = _place_replicated_module(Qwen4ExpGatedResidual(p0_config), config)
        self.mlp_hyper_connection = _place_replicated_module(Qwen4ExpGatedResidual(p0_config), config)

        if layer_type == "linear_attention":
            self.linear_attn = _place_replicated_module(
                Qwen3_5GatedDeltaNet(_text_config(hf_config), layer_number - 1, args=args), config
            )
            self.self_attention = None
        elif layer_type == "qwen_sparse_attention":
            self.linear_attn = None
            self.self_attention = _place_replicated_module(Qwen4ExpSparseAttentionReference(p0_config), config)
        else:
            raise ValueError(f"unsupported Qwen4-Exp layer type: {layer_type}")

        self.ple = None
        if ple_layer_index is not None:

            def embedding_factory(num_embeddings: int, embedding_dim: int):
                embedding = tensor_parallel.VocabParallelEmbedding(
                    num_embeddings,
                    embedding_dim,
                    init_method=config.init_method,
                    config=config,
                    tp_group=self.tp_group,
                )
                embedding.weight.qwen4_vocab_start_index = embedding.vocab_start_index
                embedding.weight.qwen4_vocab_end_index = embedding.vocab_end_index
                return embedding

            self.ple = Qwen4ExpPLE(p0_config, ple_layer_index, embedding_factory)
            self.ple = _place_replicated_module(self.ple, config)

        self.mlp = build_module(mlp_spec, config=config, pg_collection=pg_collection)
        if hasattr(self.mlp, "set_layer_number"):
            self.mlp.set_layer_number(layer_number)

    def get_qkv_layer_norm_weights(self):
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        padding_mask=None,
        **kwargs,
    ):
        del (
            attention_mask,
            context_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            sequence_len_offset,
            kwargs,
        )
        if inference_context is not None:
            raise ValueError("Qwen4-Exp Megatron P0 supports training and teacher-forced logprob forward")
        if packed_seq_params is None:
            raise ValueError("Qwen4-Exp requires explicit packed sequence metadata")
        packed_layout = getattr(packed_seq_params, "qwen4_exp_layout", None)
        input_ids = getattr(packed_seq_params, "qwen4_exp_input_ids", None)
        if packed_layout is None or input_ids is None:
            raise ValueError("Qwen4-Exp packed metadata was not bound by Qwen4ExpGPTModel")

        hidden_states = _gather_sequence_parallel(hidden_states, self.config, self.tp_group)
        if hidden_states.shape[-1] == self.p0_config.hidden_size:
            hidden_states = hidden_states.repeat(1, 1, self.p0_config.hyper_connection_count)
        if hidden_states.shape[-1] != self.p0_config.hyper_connection_width:
            raise ValueError("Qwen4-Exp decoder received an invalid mHC stream width")
        if hidden_states.shape[0] != packed_layout.positions.numel():
            raise ValueError("Qwen4-Exp gathered sequence length disagrees with packed metadata")

        if self.ple is not None:
            hidden_states = hidden_states + self.ple(hidden_states, input_ids, packed_layout)

        mixed, residual, injection = self.attn_hyper_connection(hidden_states)
        if self.linear_attn is not None:
            block_output = self.linear_attn(
                mixed.permute(1, 0, 2),
                cu_seqlens=packed_layout.cu_seqlens,
            ).permute(1, 0, 2)
        else:
            block_output, _ = self.self_attention(mixed, packed_layout)
        hidden_states = inject_residual(block_output, residual, injection)

        mixed, residual, injection = self.mlp_hyper_connection(hidden_states)
        if self.config.sequence_parallel:
            widths = (mixed.shape[-1], residual.shape[-1], injection.shape[-1])
            packed = torch.cat((mixed, residual, injection), dim=-1)
            packed = _scatter_sequence_parallel(packed, self.config, self.tp_group)
            mixed, residual, injection = torch.split(packed, widths, dim=-1)
            if padding_mask is not None:
                raise ValueError("Qwen4-Exp P0 does not accept a separate MoE padding mask")
        mlp_output = self.mlp(mixed, padding_mask=padding_mask)
        if isinstance(mlp_output, tuple):
            mlp_output, mlp_bias = mlp_output
            if mlp_bias is not None:
                mlp_output = mlp_output + mlp_bias
        return inject_residual(mlp_output, residual, injection), context


class Qwen4ExpGPTModel(GPTModel):
    """GPT shell that binds packed token metadata to every Qwen4-Exp layer."""

    def __init__(self, *args, p0_config: Qwen4ExpP0Config, **kwargs):
        super().__init__(*args, **kwargs)
        self.qwen4_exp_config = p0_config
        if self.decoder.final_layernorm is None:
            raise ValueError("Qwen4-Exp requires a final decoder stage")
        self.decoder.final_layernorm = _place_replicated_module(
            Qwen4ExpGatedResidual(p0_config, use_combine=False), self.config
        )

    def forward(self, input_ids, position_ids, attention_mask, *args, packed_seq_params=None, **kwargs):
        if packed_seq_params is None:
            raise ValueError("Qwen4-Exp requires packed_seq_params")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Qwen4-Exp P0 requires a single flat packed token row")
        layout = Qwen4ExpPackedLayout.from_cu_seqlens(input_ids.numel(), packed_seq_params.cu_seqlens_q)
        packed_seq_params.qwen4_exp_input_ids = input_ids.reshape(-1)
        packed_seq_params.qwen4_exp_layout = layout
        return super().forward(
            input_ids,
            position_ids,
            attention_mask,
            *args,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )


def _validate_p0_runtime(args, p0_config: Qwen4ExpP0Config) -> None:
    if args.pipeline_model_parallel_size != 1:
        raise ValueError("Qwen4-Exp P0 requires pipeline_model_parallel_size=1")
    if args.context_parallel_size != 1:
        raise ValueError("Qwen4-Exp P0 requires context_parallel_size=1")
    if getattr(args, "mtp_num_layers", 0) or getattr(args, "enable_mtp_training", False):
        raise ValueError("Qwen4-Exp P0 excludes MTP training")
    if args.tensor_model_parallel_size > 1 and not args.sequence_parallel:
        raise ValueError("Qwen4-Exp TP training requires sequence parallelism")
    if p0_config.hidden_size % args.tensor_model_parallel_size:
        raise ValueError("Qwen4-Exp hidden_size must be divisible by tensor_model_parallel_size")
    if not getattr(args, "moe_grouped_gemm", False):
        raise ValueError("Qwen4-Exp P0 requires moe_grouped_gemm for global expert-id mapping")
    if getattr(args, "expert_tensor_parallel_size", 1) != 1:
        raise ValueError("Qwen4-Exp P0 requires expert_tensor_parallel_size=1")
    expert_parallel_size = getattr(args, "expert_model_parallel_size", 1)
    if p0_config.num_experts % expert_parallel_size:
        raise ValueError("Qwen4-Exp num_experts must be divisible by expert_model_parallel_size")
    if getattr(args, "moe_token_dispatcher_type", "alltoall") != "alltoall":
        raise ValueError("Qwen4-Exp P0 requires the alltoall MoE token dispatcher")
    if args.num_experts != p0_config.num_experts:
        raise ValueError(f"Qwen4-Exp requires num_experts={p0_config.num_experts}")
    if args.moe_router_topk != p0_config.num_experts_per_tok:
        raise ValueError(f"Qwen4-Exp requires moe_router_topk={p0_config.num_experts_per_tok}")
    expected_pre_softmax = not p0_config.norm_topk_prob
    if getattr(args, "moe_router_pre_softmax", False) != expected_pre_softmax:
        raise ValueError(
            "Qwen4-Exp router normalization requires "
            f"moe_router_pre_softmax={expected_pre_softmax}"
        )
    if not args.untie_embeddings_and_output_weights:
        raise ValueError("Qwen4-Exp uses independent token embedding and LM head weights")


def get_qwen4_exp_model_provider(args, config, vp_stage):
    """Return a full model provider for ``--spec ...qwen4_exp.model``."""

    if vp_stage is not None:
        raise ValueError("Qwen4-Exp P0 excludes virtual pipeline parallelism")
    hf_config = _load_hf_config(args.hf_checkpoint)
    p0_config = Qwen4ExpP0Config.from_hf_config(hf_config)
    _validate_p0_runtime(args, p0_config)

    block_spec = copy.deepcopy(
        get_gpt_decoder_block_spec(
            config,
            use_transformer_engine=args.transformer_impl == "transformer_engine",
        )
    )
    layer_specs = []
    for layer_idx, layer_type in enumerate(p0_config.layer_types):
        base_layer_spec = block_spec.layer_specs[layer_idx]
        ple_layer_index = (
            p0_config.ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in p0_config.ple_layer_ids else None
        )
        layer_specs.append(
            ModuleSpec(
                module=Qwen4ExpTransformerLayer,
                params={
                    "args": args,
                    "hf_config": hf_config,
                    "p0_config": p0_config,
                    "layer_type": layer_type,
                    "mlp_spec": base_layer_spec.submodules.mlp,
                    "ple_layer_index": ple_layer_index,
                },
            )
        )
    block_spec.layer_specs = layer_specs

    def model_provider(pre_process=True, post_process=True, vp_stage=None):
        if not pre_process or not post_process or vp_stage is not None:
            raise ValueError("Qwen4-Exp P0 builds one non-pipelined model chunk")
        return Qwen4ExpGPTModel(
            config=config,
            transformer_layer_spec=block_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=True,
            post_process=True,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type="none",
            scatter_embedding_sequence_parallel=True,
            p0_config=p0_config,
        )

    return model_provider
