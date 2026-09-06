"""Executable tensor semantics for the Qwen4-Exp P0 training graph."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen4ExpP0Config


@dataclass(frozen=True)
class Qwen4ExpPackedLayout:
    """Explicit sample boundaries and reset positions for a flat token stream."""

    cu_seqlens: torch.Tensor
    positions: torch.Tensor
    max_seqlen: int
    boundaries: tuple[int, ...]

    @classmethod
    def from_cu_seqlens(cls, total_tokens: int, cu_seqlens: torch.Tensor) -> "Qwen4ExpPackedLayout":
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be a one-dimensional tensor with at least two entries")
        cu_seqlens = cu_seqlens.to(dtype=torch.long)
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if int(cu_seqlens[0].item()) != 0 or int(cu_seqlens[-1].item()) != total_tokens:
            raise ValueError(f"cu_seqlens must cover exactly {total_tokens} tokens")
        if bool(torch.any(lengths <= 0).item()):
            raise ValueError("packed samples must have positive lengths")

        starts = torch.repeat_interleave(cu_seqlens[:-1], lengths)
        positions = torch.arange(total_tokens, device=cu_seqlens.device, dtype=torch.long) - starts
        return cls(
            cu_seqlens=cu_seqlens,
            positions=positions,
            max_seqlen=int(lengths.max().item()),
            boundaries=tuple(int(value) for value in cu_seqlens.detach().cpu().tolist()),
        )

    def slices(self):
        return zip(self.boundaries[:-1], self.boundaries[1:])


class _Qwen4ExpGroupedRMSNorm(nn.Module):
    """Qwen4-Exp zero-centered grouped RMSNorm."""

    def __init__(self, hidden_size: int, group_size: int, eps: float):
        super().__init__()
        if hidden_size % group_size:
            raise ValueError("hidden_size must be divisible by group_size")
        self.group_size = group_size
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] % self.group_size:
            raise ValueError("the hidden dimension must be divisible by group_size")
        if self.weight.shape != (hidden_states.shape[-1],):
            raise ValueError("RMSNorm weight shape must match the hidden dimension")
        grouped = hidden_states.float().unflatten(-1, (-1, self.group_size))
        normalized = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = normalized.flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(hidden_states.dtype)


class Qwen4ExpGatedResidual(nn.Module):
    """mHC stream mixer and residual-injection gate."""

    def __init__(self, config: Qwen4ExpP0Config, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hyper_connection_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = config.hyper_connection_width
        self.hc_norm = _Qwen4ExpGroupedRMSNorm(hc_hidden_size, config.hidden_size, config.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_hidden_size, bias=False)
        self.block_inject_weight = nn.Linear(hc_hidden_size, self.hc_count, bias=False) if use_combine else None

    def forward(self, hyper_input: torch.Tensor):
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError("invalid hyper-connection width")
        normalized = self.hc_norm(hyper_input)
        mix = F.silu(self.input_mix_weight_down(normalized) / self.hc_count)
        mix = torch.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = (mix * normalized.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed_input
        injection = 2.0 * torch.sigmoid(self.block_inject_weight(normalized) / self.hc_count)
        return mixed_input, hyper_input, injection


def inject_residual(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    injection_weights: torch.Tensor,
) -> torch.Tensor:
    """Write an H-wide block output into the mHC residual streams."""

    if residual.shape[:-1] != block_output.shape[:-1] or residual.shape[:-1] != injection_weights.shape[:-1]:
        raise ValueError("block output, residual, and injection weights must share leading dimensions")
    hc_count = injection_weights.shape[-1]
    if residual.shape[-1] != block_output.shape[-1] * hc_count:
        raise ValueError("residual width must equal hc_count * block output width")
    streams = residual.unflatten(-1, (hc_count, block_output.shape[-1]))
    return (streams + injection_weights.unsqueeze(-1) * block_output.unsqueeze(-2)).flatten(-2)


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(
    unigram_vocab_size: int,
    ngram_size: int,
    ple_layer_index: int,
    seed: int,
) -> torch.Tensor:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


@dataclass(frozen=True)
class Qwen4ExpNGramLayout:
    multipliers: torch.Tensor
    head_vocab_sizes: torch.Tensor
    head_offsets: torch.Tensor
    padded_vocab_size: int

    @classmethod
    def from_config(cls, config: Qwen4ExpP0Config, ple_layer_index: int = 0) -> "Qwen4ExpNGramLayout":
        sizes = []
        offsets = []
        total = 0
        for head_idx in range(config.ple_num_heads):
            global_head_idx = ple_layer_index * config.ple_num_heads + head_idx
            size = _find_nth_prime_after(config.ngram_vocab_size_base - 1, global_head_idx + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(total / divisor) * divisor
        return cls(
            multipliers=_build_layer_multipliers(
                config.vocab_size, config.ngram_size, ple_layer_index, config.seed
            ),
            head_vocab_sizes=torch.tensor(sizes, dtype=torch.long),
            head_offsets=torch.tensor(offsets, dtype=torch.long),
            padded_vocab_size=padded_vocab_size,
        )


class _Qwen4ExpNGramEmbedding(nn.Module):
    """Small/reference PLE table with checkpoint-compatible buffer names."""

    def __init__(
        self,
        config: Qwen4ExpP0Config,
        ple_layer_index: int = 0,
        embedding_factory: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()
        self.config = config
        layout = Qwen4ExpNGramLayout.from_config(config, ple_layer_index)
        self.register_buffer("layer_multipliers", layout.multipliers)
        self.register_buffer("ngram_heads_vocab_sizes", layout.head_vocab_sizes)
        self.register_buffer("ngram_heads_offsets", layout.head_offsets)
        num_embeddings = layout.padded_vocab_size
        embedding_dim = config.ple_embed_dim // config.ple_num_heads
        self.ngram_embedding = (
            nn.Embedding(num_embeddings, embedding_dim)
            if embedding_factory is None
            else embedding_factory(num_embeddings, embedding_dim)
        )
        self.ngram_embedding.weight.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, packed_layout: Qwen4ExpPackedLayout) -> torch.Tensor:
        """Hash n-grams from registered buffers, resetting history at EOS/sample starts."""

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel() != packed_layout.positions.numel():
            raise ValueError("input_ids and packed layout have different token counts")
        device = flat_ids.device
        multipliers = self.layer_multipliers.to(device)
        head_vocab_sizes = self.ngram_heads_vocab_sizes.to(device)
        head_offsets = self.ngram_heads_offsets.to(device)

        token_positions = torch.arange(flat_ids.numel(), device=device, dtype=torch.long)
        sample_starts = token_positions - packed_layout.positions.to(device)
        eos_positions = torch.where(flat_ids == self.config.eos_token_id, token_positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=0).values
        previous_eos = torch.cat((eos_positions.new_full((1,), -1), previous_eos_inclusive[:-1]))
        lexical_segment_starts = torch.maximum(sample_starts, previous_eos + 1)

        shifted_tokens = [flat_ids]
        for shift in range(1, self.config.ngram_size):
            source_positions = token_positions - shift
            valid = token_positions - lexical_segment_starts >= shift
            shifted = flat_ids.new_full(flat_ids.shape, self.config.eos_token_id)
            shifted[valid] = flat_ids[source_positions[valid]]
            shifted_tokens.append(shifted)

        blocks = []
        for ngram in range(2, self.config.ngram_size + 1):
            start_idx = (ngram - 2) * self.config.heads_per_ngram
            end_idx = start_idx + self.config.heads_per_ngram
            mixed_ids = shifted_tokens[0] * multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(mixed_ids, shifted_tokens[position] * multipliers[position])
            sizes = head_vocab_sizes[start_idx:end_idx]
            offsets = head_offsets[start_idx:end_idx]
            blocks.append(torch.remainder(mixed_ids.unsqueeze(-1), sizes) + offsets)
        ngram_ids = torch.cat(blocks, dim=-1)
        return self.ngram_embedding(ngram_ids).flatten(-2)


class Qwen4ExpPLE(nn.Module):
    """Packed-sequence PLE reference; convolution state resets at each sample."""

    def __init__(
        self,
        config: Qwen4ExpP0Config,
        ple_layer_index: int = 0,
        embedding_factory: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()
        self.config = config
        self.ple_embedding = _Qwen4ExpNGramEmbedding(config, ple_layer_index, embedding_factory)
        self.key_proj = nn.Linear(config.ple_embed_dim, config.hyper_connection_width, bias=False)
        self.value_proj = nn.Linear(config.ple_embed_dim, config.hidden_size, bias=False)
        self.norm_key = _Qwen4ExpGroupedRMSNorm(
            config.hyper_connection_width, config.hidden_size, config.rms_norm_eps
        )
        self.norm_query = _Qwen4ExpGroupedRMSNorm(
            config.hyper_connection_width, config.hidden_size, config.rms_norm_eps
        )
        self.norm_conv = _Qwen4ExpGroupedRMSNorm(
            config.hyper_connection_width, config.hidden_size, config.rms_norm_eps
        )
        self.conv1d = nn.Conv1d(
            config.hyper_connection_width,
            config.hyper_connection_width,
            kernel_size=config.ple_conv_kernel_size,
            dilation=config.ngram_size,
            groups=config.hyper_connection_width,
            bias=False,
        )

    def _packed_short_conv(
        self,
        hidden_states: torch.Tensor,
        packed_layout: Qwen4ExpPackedLayout,
    ) -> torch.Tensor:
        state_len = (self.config.ple_conv_kernel_size - 1) * self.config.ngram_size
        outputs = []
        for start, end in packed_layout.slices():
            segment = hidden_states[start:end].transpose(0, 1).unsqueeze(0)
            segment = F.pad(segment, (state_len, 0))
            outputs.append(F.silu(self.conv1d(segment)).squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        packed_layout: Qwen4ExpPackedLayout,
    ) -> torch.Tensor:
        if hidden_states.ndim == 3:
            if hidden_states.shape[1] != 1:
                raise ValueError("the P0 packed reference accepts batch size one")
            flat_hidden = hidden_states[:, 0]
        elif hidden_states.ndim == 2:
            flat_hidden = hidden_states
        else:
            raise ValueError("hidden_states must have shape [S, H*hc] or [S, 1, H*hc]")

        embeddings = self.ple_embedding(input_ids, packed_layout)
        key = self.norm_key(self.key_proj(embeddings)).unflatten(
            -1, (self.config.hyper_connection_count, self.config.hidden_size)
        )
        value = self.value_proj(embeddings)
        query = self.norm_query(flat_hidden).unflatten(
            -1, (self.config.hyper_connection_count, self.config.hidden_size)
        )
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.config.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
        output = gated_value + self._packed_short_conv(self.norm_conv(gated_value), packed_layout)
        return output.unsqueeze(1) if hidden_states.ndim == 3 else output


def _build_rope_cos_sin(
    positions: torch.Tensor,
    rotary_dim: int,
    rope_theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, rotary_dim, 2, device=positions.device, dtype=torch.float32) / rotary_dim)
    )
    frequencies = torch.outer(positions.float(), inv_freq)
    embeddings = torch.cat((frequencies, frequencies), dim=-1)
    return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_partial_rope(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    rotated = rotary * cos + _rotate_half(rotary) * sin
    return torch.cat((rotated, passthrough), dim=-1)


class _Qwen4ExpQSAIndexer(nn.Module):
    """Frozen checkpoint state retained for SGLang parity during P0."""

    def __init__(self, config: Qwen4ExpP0Config):
        super().__init__()
        projection_size = (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim
        self.index_qk_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.q_layernorm = _Qwen4ExpGroupedRMSNorm(
            config.indexer_head_dim, config.indexer_head_dim, config.rms_norm_eps
        )
        self.k_layernorm = _Qwen4ExpGroupedRMSNorm(
            config.indexer_head_dim, config.indexer_head_dim, config.rms_norm_eps
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)


def _packed_dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    packed_layout: Qwen4ExpPackedLayout,
) -> torch.Tensor:
    """Run the exact full-selection QSA case with PyTorch's fused causal SDPA."""

    enable_gqa = query.shape[1] != key.shape[1]
    outputs = []
    for start, end in packed_layout.slices():
        segment_query = query[start:end].transpose(0, 1).unsqueeze(0)
        segment_key = key[start:end].transpose(0, 1).unsqueeze(0)
        segment_value = value[start:end].transpose(0, 1).unsqueeze(0)
        output = F.scaled_dot_product_attention(
            segment_query,
            segment_key,
            segment_value,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=enable_gqa,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


class Qwen4ExpQSA(nn.Module):
    """Full-selection QSA training path for the P0 context-length contract."""

    def __init__(self, config: Qwen4ExpP0Config):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, 2 * config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = _Qwen4ExpGroupedRMSNorm(config.head_dim, config.head_dim, config.rms_norm_eps)
        self.k_norm = _Qwen4ExpGroupedRMSNorm(config.head_dim, config.head_dim, config.rms_norm_eps)
        self.indexer = _Qwen4ExpQSAIndexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        packed_layout: Qwen4ExpPackedLayout,
    ) -> torch.Tensor:
        if packed_layout.max_seqlen > self.config.indexer_budget:
            raise ValueError(
                "Qwen4-Exp P0 requires every packed sequence to fit within "
                f"the {self.config.indexer_budget}-token QSA budget"
            )
        flat_hidden = hidden_states[:, 0] if hidden_states.ndim == 3 else hidden_states
        rotary_dim = int(self.config.head_dim * self.config.partial_rotary_factor)
        cos, sin = _build_rope_cos_sin(
            packed_layout.positions.to(flat_hidden.device), rotary_dim, self.config.rope_theta, flat_hidden.dtype
        )

        projected_query = self.q_proj(flat_hidden).unflatten(
            -1, (self.config.num_attention_heads, 2 * self.config.head_dim)
        )
        query, gate = projected_query.chunk(2, dim=-1)
        key = self.k_proj(flat_hidden).unflatten(-1, (self.config.num_key_value_heads, self.config.head_dim))
        value = self.v_proj(flat_hidden).unflatten(-1, (self.config.num_key_value_heads, self.config.head_dim))
        query = _apply_partial_rope(self.q_norm(query), cos, sin)
        key = _apply_partial_rope(self.k_norm(key), cos, sin)
        attention_output = _packed_dense_attention(query, key, value, packed_layout)
        attention_output = attention_output * torch.sigmoid(gate)
        output = self.o_proj(attention_output.flatten(-2))
        if hidden_states.ndim == 3:
            output = output.unsqueeze(1)
        return output
