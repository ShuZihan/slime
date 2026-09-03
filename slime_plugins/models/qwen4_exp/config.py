"""Validated Qwen4-Exp text-model configuration used by the P0 RL path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_tuple(value: Iterable[Any]) -> Tuple[Any, ...]:
    return tuple(value)


def _normalize_layer_type(layer_type: str) -> str:
    aliases = {
        "full_attention": "qwen_sparse_attention",
        "linear_attention": "linear_attention",
        "qwen_sparse_attention": "qwen_sparse_attention",
    }
    try:
        return aliases[layer_type]
    except KeyError as exc:
        raise ValueError(f"unsupported Qwen4-Exp layer type: {layer_type!r}") from exc


@dataclass(frozen=True)
class Qwen4ExpP0Config:
    """Architecture fields whose semantics affect the P0 training graph.

    The public checkpoint is a conditional-generation model.  P0 consumes the
    nested text configuration and deliberately leaves vision and MTP outside
    the graph.  PLE layer numbers retain the checkpoint's one-based convention.
    """

    hidden_size: int
    num_hidden_layers: int
    layer_types: Tuple[str, ...]
    rms_norm_eps: float

    hyper_connection_count: int
    hc_lowrank: int

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    partial_rotary_factor: float
    rope_theta: float

    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int

    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int

    ple_layer_ids: Tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_ngram_vocab_size_divisible_by: int
    seed: int
    split_ngram_parts: int

    vocab_size: int
    eos_token_id: int
    pad_token_id: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    norm_topk_prob: bool
    hidden_act: str
    output_gate_type: str
    attention_bias: bool
    tie_word_embeddings: bool

    @property
    def hyper_connection_width(self) -> int:
        return self.hidden_size * self.hyper_connection_count

    @property
    def index_block_topk(self) -> int:
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
        return self.indexer_budget // self.indexer_compress_ratio

    @property
    def ple_num_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @classmethod
    def from_hf_config(cls, config: Any) -> "Qwen4ExpP0Config":
        text_config = _get(config, "text_config", config)
        layer_types = _get(text_config, "layer_types")
        if layer_types is None:
            pattern = _get(text_config, "full_attention_interval", 4)
            layer_types = [
                "qwen_sparse_attention" if (layer_idx + 1) % pattern == 0 else "linear_attention"
                for layer_idx in range(_get(text_config, "num_hidden_layers"))
            ]

        eos_token_id = _get(text_config, "eos_token_id", _get(config, "eos_token_id"))
        if isinstance(eos_token_id, Sequence) and not isinstance(eos_token_id, (str, bytes)):
            if not eos_token_id:
                raise ValueError("eos_token_id cannot be empty")
            eos_token_id = eos_token_id[0]
        if eos_token_id is None:
            raise ValueError("Qwen4-Exp requires eos_token_id for packed-sequence boundaries")

        pad_token_id = _get(text_config, "pad_token_id", _get(config, "pad_token_id"))
        if pad_token_id is None:
            pad_token_id = eos_token_id

        rope_parameters = _get(text_config, "rope_parameters", {}) or {}
        partial_rotary_factor = _get(text_config, "partial_rotary_factor")
        if partial_rotary_factor is None:
            partial_rotary_factor = _get(rope_parameters, "partial_rotary_factor", 1.0)
        rope_theta = _get(text_config, "rope_theta")
        if rope_theta is None:
            rope_theta = _get(rope_parameters, "rope_theta", 10_000.0)
        seed = _get(text_config, "seed")
        if seed is None:
            seed = 1234
        hidden_act = str(_get(text_config, "hidden_act", "silu"))
        result = cls(
            hidden_size=int(_get(text_config, "hidden_size")),
            num_hidden_layers=int(_get(text_config, "num_hidden_layers")),
            layer_types=_as_tuple(_normalize_layer_type(item) for item in layer_types),
            rms_norm_eps=float(_get(text_config, "rms_norm_eps")),
            hyper_connection_count=int(_get(text_config, "hc_count")),
            hc_lowrank=int(_get(text_config, "hc_lowrank")),
            num_attention_heads=int(_get(text_config, "num_attention_heads")),
            num_key_value_heads=int(_get(text_config, "num_key_value_heads")),
            head_dim=int(_get(text_config, "head_dim")),
            partial_rotary_factor=float(partial_rotary_factor),
            rope_theta=float(rope_theta),
            linear_num_key_heads=int(_get(text_config, "linear_num_key_heads")),
            linear_num_value_heads=int(_get(text_config, "linear_num_value_heads")),
            linear_key_head_dim=int(_get(text_config, "linear_key_head_dim")),
            linear_value_head_dim=int(_get(text_config, "linear_value_head_dim")),
            linear_conv_kernel_dim=int(_get(text_config, "linear_conv_kernel_dim")),
            indexer_n_heads=int(_get(text_config, "indexer_n_heads")),
            indexer_kv_heads=int(_get(text_config, "indexer_kv_heads")),
            indexer_head_dim=int(_get(text_config, "indexer_head_dim")),
            indexer_budget=int(_get(text_config, "indexer_budget")),
            indexer_compress_ratio=int(_get(text_config, "indexer_compress_ratio")),
            ple_layer_ids=_as_tuple(int(item) for item in _get(text_config, "ple_layer_ids", ())),
            ple_embed_dim=int(_get(text_config, "ple_embed_dim")),
            ple_conv_kernel_size=int(_get(text_config, "ple_conv_kernel_size", 4)),
            ngram_size=int(_get(text_config, "ngram_size", 3)),
            heads_per_ngram=int(_get(text_config, "heads_per_ngram", 8)),
            ngram_vocab_size_base=int(_get(text_config, "ngram_vocab_size_base", 20_000_000)),
            make_ngram_vocab_size_divisible_by=int(
                _get(text_config, "make_ngram_vocab_size_divisible_by", 128)
            ),
            seed=int(seed),
            split_ngram_parts=int(_get(text_config, "split_ngram_parts", 512)),
            vocab_size=int(_get(text_config, "vocab_size")),
            eos_token_id=int(eos_token_id),
            pad_token_id=int(pad_token_id),
            num_experts=int(_get(text_config, "num_experts")),
            num_experts_per_tok=int(_get(text_config, "num_experts_per_tok")),
            moe_intermediate_size=int(_get(text_config, "moe_intermediate_size")),
            shared_expert_intermediate_size=int(_get(text_config, "shared_expert_intermediate_size")),
            norm_topk_prob=bool(_get(text_config, "norm_topk_prob", True)),
            hidden_act=hidden_act,
            output_gate_type=str(_get(text_config, "output_gate_type") or hidden_act),
            attention_bias=bool(_get(text_config, "attention_bias", False)),
            tie_word_embeddings=bool(
                _get(text_config, "tie_word_embeddings", _get(config, "tie_word_embeddings", False))
            ),
        )
        result.validate()
        return result

    def validate_public_release_contract(self) -> None:
        """Validate the fixed public Qwen4-Exp checkpoint architecture used by P0."""

        expected = {
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "hyper_connection_count": 4,
            "hc_lowrank": 320,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000.0,
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
            "ple_layer_ids": (2,),
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 20_000_000,
            "make_ngram_vocab_size_divisible_by": 128,
            "seed": 1234,
            "split_ngram_parts": 128,
            "vocab_size": 248_320,
            "eos_token_id": 248_044,
            "pad_token_id": 248_044,
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
        mismatches = {
            name: {"actual": getattr(self, name), "expected": expected_value}
            for name, expected_value in expected.items()
            if getattr(self, name) != expected_value
        }
        expected_layers = tuple(
            "qwen_sparse_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention"
            for layer_idx in range(48)
        )
        if self.layer_types != expected_layers:
            mismatches["layer_types"] = {
                "actual": self.layer_types,
                "expected": expected_layers,
            }
        if mismatches:
            raise ValueError(f"checkpoint does not match the Qwen4-Exp public release: {mismatches}")

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.num_hidden_layers <= 0:
            raise ValueError("hidden_size and num_hidden_layers must be positive")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries for {self.num_hidden_layers} transformer layers"
            )
        if self.hyper_connection_count <= 1:
            raise ValueError("Qwen4-Exp requires multiple hyper-connection streams")
        if self.hc_lowrank <= 0:
            raise ValueError("hc_lowrank must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError("partial rotary dimension must be a positive even number")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear value heads must be divisible by linear key heads")
        if self.indexer_kv_heads != 1:
            raise ValueError("the Qwen4-Exp P0 QSA reference requires one indexer KV head")
        if self.indexer_budget <= 0 or self.indexer_compress_ratio <= 0:
            raise ValueError("QSA budget and selection block size must be positive")
        _ = self.index_block_topk
        if self.ngram_size <= 1 or self.heads_per_ngram <= 0:
            raise ValueError("PLE ngram size and heads per ngram must be positive")
        if self.ple_embed_dim % self.ple_num_heads:
            raise ValueError("ple_embed_dim must be divisible by the number of PLE heads")
        if self.ngram_vocab_size_base <= 0 or self.make_ngram_vocab_size_divisible_by <= 0:
            raise ValueError("PLE vocabulary size and divisibility factor must be positive")
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")
        if len(set(self.ple_layer_ids)) != len(self.ple_layer_ids):
            raise ValueError("ple_layer_ids contains duplicates")
        if any(layer_id < 1 or layer_id > self.num_hidden_layers for layer_id in self.ple_layer_ids):
            raise ValueError("PLE layer ids use one-based transformer-layer numbering")
        if any(self.layer_types[layer_id - 1] != "linear_attention" for layer_id in self.ple_layer_ids):
            raise ValueError("PLE is supported only on linear-attention layers")
        if self.num_experts <= 0 or self.num_experts_per_tok <= 0:
            raise ValueError("Qwen4-Exp P0 requires its MoE decoder")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok exceeds num_experts")
        if self.hidden_act != "silu":
            raise ValueError("Qwen4-Exp P0 requires silu expert activation")
        if self.output_gate_type not in {"sigmoid", "silu"}:
            raise ValueError("unsupported Qwen4-Exp GDN output gate activation")
        if self.attention_bias:
            raise ValueError("Qwen4-Exp P0 requires bias-free attention projections")
        if self.tie_word_embeddings:
            raise ValueError("Qwen4-Exp P0 requires independent embedding and LM head weights")
