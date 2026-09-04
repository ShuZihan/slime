from __future__ import annotations

import re

import torch

from slime_plugins.models.qwen4_exp.config import Qwen4ExpP0Config
from slime_plugins.models.qwen4_exp.reference import Qwen4ExpNGramLayout

from .common import SafetensorReader, strip_mcore_wrappers


def _merge_gated_qkv(reader: SafetensorReader, prefix: str, config) -> torch.Tensor:
    """Pack checkpoint q/gate, k, and v into MCore's per-KV-group layout."""

    query_gate = reader.get_tensor(f"{prefix}.q_proj.weight")
    key = reader.get_tensor(f"{prefix}.k_proj.weight")
    value = reader.get_tensor(f"{prefix}.v_proj.weight")
    num_groups = config.num_key_value_heads
    queries_per_group = config.num_attention_heads // num_groups
    head_dim = config.head_dim
    trailing_shape = query_gate.shape[1:]
    query_gate = query_gate.reshape(num_groups, queries_per_group, 2, head_dim, *trailing_shape)
    query_gate = query_gate.transpose(1, 2).flatten(1, 3)
    key = key.reshape(num_groups, head_dim, *trailing_shape)
    value = value.reshape(num_groups, head_dim, *trailing_shape)
    return torch.cat((query_gate, key, value), dim=1).reshape(-1, *trailing_shape).contiguous()


def _read_expert(reader: SafetensorReader, name: str, expert_idx: int) -> torch.Tensor:
    shape = reader.get_shape(name)
    if not shape or not 0 <= expert_idx < shape[0]:
        raise IndexError(f"expert {expert_idx} is outside {name} with shape {shape}")
    get_tensor_slice = getattr(reader, "get_tensor_slice", None)
    if get_tensor_slice is not None:
        return get_tensor_slice(name, expert_idx).contiguous()
    return reader.get_tensor(name)[expert_idx].contiguous()


class Qwen4ExpHfLoader:
    """Map public Qwen4-Exp tensors and stream the frozen PLE table by row overlap."""

    _PLE_PARAMETER = "ple.ple_embedding.ngram_embedding.weight"
    _PLE_SHARD_PATTERN = re.compile(r"\.ngram_embedding\.shard_(\d+)\.weight$")
    _PLE_DERIVED_BUFFERS = {
        "layer_multipliers",
        "ngram_heads_vocab_sizes",
        "ngram_heads_offsets",
    }

    def __call__(self, name: str, reader: SafetensorReader, hf_config) -> torch.Tensor:
        name = strip_mcore_wrappers(name).removeprefix("language_model.")
        direct_mapping = {
            "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
        }
        if name in direct_mapping:
            return reader.get_tensor(direct_mapping[name])
        if name.startswith("decoder.final_layernorm."):
            suffix = name.removeprefix("decoder.final_layernorm.")
            return reader.get_tensor(f"model.language_model.hyper_connection_mixer.{suffix}")
        if name.startswith("hyper_connection_mixer."):
            return reader.get_tensor(f"model.language_model.{name}")

        layer_match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
        if not layer_match:
            raise KeyError(f"unsupported Qwen4-Exp Megatron parameter {name!r}")
        layer_idx, rest = layer_match.groups()
        prefix = f"model.language_model.layers.{layer_idx}"
        config = getattr(hf_config, "text_config", hf_config)

        for direct_prefix in ("attn_hyper_connection.", "mlp_hyper_connection.", "linear_attn.", "ple."):
            if rest.startswith(direct_prefix) and rest != self._PLE_PARAMETER:
                return reader.get_tensor(f"{prefix}.{rest}")

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
            return _merge_gated_qkv(reader, f"{prefix}.self_attn", config)
        if rest in attention_mapping:
            return reader.get_tensor(f"{prefix}.{attention_mapping[rest]}")

        if rest == "mlp.router.weight":
            return reader.get_tensor(f"{prefix}.mlp.gate.weight")
        expert_match = re.fullmatch(r"mlp\.experts\.linear_fc([12])(?:\.weight)?(\d+)?", rest)
        if expert_match:
            projection, expert_idx = expert_match.groups()
            suffix = "gate_up_proj" if projection == "1" else "down_proj"
            source_name = f"{prefix}.mlp.experts.{suffix}"
            return reader.get_tensor(source_name) if expert_idx is None else _read_expert(
                reader, source_name, int(expert_idx)
            )
        local_expert_match = re.fullmatch(r"mlp\.experts\.local_experts\.(\d+)\.linear_fc([12])\.weight", rest)
        if local_expert_match:
            expert_idx, projection = local_expert_match.groups()
            suffix = "gate_up_proj" if projection == "1" else "down_proj"
            return _read_expert(reader, f"{prefix}.mlp.experts.{suffix}", int(expert_idx))

        shared_mapping = {
            "mlp.shared_experts.linear_fc1.weight": ("shared_expert.gate_proj.weight", "shared_expert.up_proj.weight"),
            "mlp.shared_experts.linear_fc2.weight": ("shared_expert.down_proj.weight",),
            "mlp.shared_experts.gate_weight": ("shared_expert_gate.weight",),
        }
        if rest in shared_mapping:
            tensors = [reader.get_tensor(f"{prefix}.mlp.{suffix}") for suffix in shared_mapping[rest]]
            return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        raise KeyError(f"unsupported Qwen4-Exp Megatron parameter {name!r}")

    def load_parameter(
        self,
        args,
        name: str,
        parameter: torch.Tensor,
        reader: SafetensorReader,
        hf_config,
    ) -> bool:
        canonical_name = strip_mcore_wrappers(name).removeprefix("language_model.")
        layer_match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", canonical_name)
        if not layer_match or layer_match.group(2) != self._PLE_PARAMETER:
            return False
        layer_idx = int(layer_match.group(1))
        p0_config = Qwen4ExpP0Config.from_hf_config(hf_config)
        try:
            ple_layer_index = p0_config.ple_layer_ids.index(layer_idx + 1)
        except ValueError as exc:
            raise ValueError(f"Megatron exposes PLE table on non-PLE layer {layer_idx}") from exc
        layout = Qwen4ExpNGramLayout.from_config(p0_config, ple_layer_index)

        global_rows = layout.padded_vocab_size
        local_start = int(getattr(parameter, "qwen4_vocab_start_index", 0))
        local_end = int(getattr(parameter, "qwen4_vocab_end_index", local_start + parameter.shape[0]))
        if local_end - local_start != parameter.shape[0] or not 0 <= local_start < local_end <= global_rows:
            raise ValueError(
                f"invalid PLE TP row interval [{local_start}, {local_end}) for {global_rows} global rows"
            )
        if parameter.shape[1] != p0_config.ple_embed_dim // p0_config.ple_num_heads:
            raise ValueError("PLE embedding width does not match Qwen4-Exp config")

        source_prefix = f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding"
        source_shards = {}
        for source_name in reader.weight_map:
            if not source_name.startswith(source_prefix):
                continue
            match = self._PLE_SHARD_PATTERN.search(source_name)
            if match:
                source_shards[int(match.group(1))] = source_name
        expected_indices = set(range(p0_config.split_ngram_parts))
        if set(source_shards) != expected_indices:
            missing = sorted(expected_indices - set(source_shards))
            extra = sorted(set(source_shards) - expected_indices)
            raise ValueError(f"PLE shard index mismatch; missing={missing}, extra={extra}")

        parameter.zero_()
        shard_rows = (global_rows + p0_config.split_ngram_parts - 1) // p0_config.split_ngram_parts
        covered_until = local_start
        for shard_idx in range(p0_config.split_ngram_parts):
            source_name = source_shards[shard_idx]
            source_shape = reader.get_shape(source_name)
            if len(source_shape) != 2 or source_shape[1] != parameter.shape[1]:
                raise ValueError(f"invalid PLE source shard shape for {source_name}: {source_shape}")
            source_start = shard_idx * shard_rows
            source_end = source_start + source_shape[0]
            overlap_start = max(source_start, local_start)
            overlap_end = min(source_end, local_end)
            if overlap_start >= overlap_end:
                continue
            source_offset = overlap_start - source_start
            target_offset = overlap_start - local_start
            row_count = overlap_end - overlap_start
            loaded = reader.get_tensor_slice(
                source_name,
                slice(source_offset, source_offset + row_count),
            )
            parameter[target_offset : target_offset + row_count].copy_(
                loaded.to(device=parameter.device, dtype=parameter.dtype)
            )
            if overlap_start != covered_until:
                raise ValueError(f"PLE TP interval has a gap before global row {overlap_start}")
            covered_until = overlap_end
        if covered_until != local_end:
            raise ValueError(f"PLE TP interval is incomplete: loaded through {covered_until}, expected {local_end}")
        return True

    def finalize_load(self, args, model, reader: SafetensorReader, hf_config, *, load_audit=None) -> None:
        """Verify config-derived PLE buffers against their checkpoint tensors."""

        del args
        modules = list(model) if isinstance(model, (list, tuple)) else [model]
        observed = set()
        for model_module in modules:
            for name, buffer in model_module.named_buffers():
                canonical_name = strip_mcore_wrappers(name).removeprefix("language_model.")
                match = re.fullmatch(
                    r"decoder\.layers\.(\d+)\.ple\.ple_embedding\.([^\.]+)", canonical_name
                )
                if match is None or match.group(2) not in self._PLE_DERIVED_BUFFERS:
                    continue
                layer_idx, buffer_name = match.groups()
                source_name = f"model.language_model.layers.{layer_idx}.ple.ple_embedding.{buffer_name}"
                expected = reader.get_tensor(source_name)
                if expected.shape != buffer.shape:
                    raise ValueError(
                        f"shape mismatch for derived PLE buffer {source_name}: "
                        f"checkpoint={tuple(expected.shape)}, Megatron={tuple(buffer.shape)}"
                    )
                expected_local = expected.to(device=buffer.device, dtype=buffer.dtype)
                if not torch.equal(buffer, expected_local):
                    raise ValueError(f"derived PLE buffer does not match checkpoint: {source_name}")
                observed.add(source_name)
                if load_audit is not None:
                    load_audit.observe(
                        name,
                        buffer,
                        expected=expected,
                        load_mode="derived_config_verified",
                    )

        config = Qwen4ExpP0Config.from_hf_config(hf_config)
        expected_count = len(config.ple_layer_ids) * len(self._PLE_DERIVED_BUFFERS)
        if len(observed) != expected_count:
            raise ValueError(f"verified {len(observed)} derived PLE buffers; expected {expected_count}")


qwen4_exp_hf_loader = Qwen4ExpHfLoader()
