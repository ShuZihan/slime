from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from slime_plugins.models.qwen4_exp.config import Qwen4ExpP0Config
from slime_plugins.models.qwen4_exp.reference import (
    Qwen4ExpGatedResidual,
    Qwen4ExpNGramLayout,
    Qwen4ExpPLE,
    Qwen4ExpPackedLayout,
    Qwen4ExpSparseAttentionReference,
    build_ngram_ids,
    build_rope_cos_sin,
    inject_residual,
    qsa_select_token_indices,
)


def tiny_config() -> Qwen4ExpP0Config:
    return Qwen4ExpP0Config(
        hidden_size=8,
        num_hidden_layers=4,
        layer_types=("linear_attention", "linear_attention", "linear_attention", "qwen_sparse_attention"),
        rms_norm_eps=1e-6,
        hyper_connection_count=2,
        hc_lowrank=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        partial_rotary_factor=0.5,
        rope_theta=10_000.0,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=4,
        indexer_compress_ratio=2,
        ple_layer_ids=(2,),
        ple_embed_dim=8,
        ple_conv_kernel_size=2,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=8,
        seed=1234,
        split_ngram_parts=4,
        vocab_size=64,
        eos_token_id=63,
        pad_token_id=63,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        norm_topk_prob=True,
        hidden_act="silu",
        output_gate_type="sigmoid",
        attention_bias=False,
        tie_word_embeddings=False,
    )


@pytest.mark.unit
def test_packed_layout_resets_positions():
    layout = Qwen4ExpPackedLayout.from_cu_seqlens(7, torch.tensor([0, 3, 7], dtype=torch.int32))

    assert layout.positions.tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert layout.sequence_ids.tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert layout.max_seqlen == 4


@pytest.mark.unit
def test_ngram_hash_has_no_cross_sample_history():
    config = tiny_config()
    ngram_layout = Qwen4ExpNGramLayout.from_config(config)
    tokens = torch.tensor([4, 5, 6, 20, 21, 22])
    packed = Qwen4ExpPackedLayout.from_cu_seqlens(6, torch.tensor([0, 3, 6]))
    left = Qwen4ExpPackedLayout.from_cu_seqlens(3, torch.tensor([0, 3]))
    right = Qwen4ExpPackedLayout.from_cu_seqlens(3, torch.tensor([0, 3]))

    packed_ids = build_ngram_ids(
        tokens,
        packed,
        ngram_layout,
        config.ngram_size,
        config.heads_per_ngram,
        config.eos_token_id,
    )
    separate_ids = torch.cat(
        [
            build_ngram_ids(
                segment,
                layout,
                ngram_layout,
                config.ngram_size,
                config.heads_per_ngram,
                config.eos_token_id,
            )
            for segment, layout in ((tokens[:3], left), (tokens[3:], right))
        ]
    )

    torch.testing.assert_close(packed_ids, separate_ids)


@pytest.mark.unit
def test_ngram_hash_resets_history_after_eos_inside_a_sample():
    config = tiny_config()
    ngram_layout = Qwen4ExpNGramLayout.from_config(config)
    tokens = torch.tensor([4, 5, config.eos_token_id, 20, 21, 22])
    packed = Qwen4ExpPackedLayout.from_cu_seqlens(6, torch.tensor([0, 6]))

    actual = build_ngram_ids(
        tokens,
        packed,
        ngram_layout,
        config.ngram_size,
        config.heads_per_ngram,
        config.eos_token_id,
    )
    right = Qwen4ExpPackedLayout.from_cu_seqlens(3, torch.tensor([0, 3]))
    expected_right = build_ngram_ids(
        tokens[3:],
        right,
        ngram_layout,
        config.ngram_size,
        config.heads_per_ngram,
        config.eos_token_id,
    )

    torch.testing.assert_close(actual[3:], expected_right)


@pytest.mark.unit
def test_gated_residual_zero_weights_have_exact_stream_formula():
    config = tiny_config()
    module = Qwen4ExpGatedResidual(config)
    for parameter in module.parameters():
        parameter.data.zero_()
    hyper_input = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 10

    mixed, residual, injection = module(hyper_input)
    normalized = module.hc_norm(hyper_input).unflatten(-1, (2, 8))

    torch.testing.assert_close(mixed, 0.5 * normalized.mean(dim=-2))
    torch.testing.assert_close(injection, torch.ones_like(injection))
    block_output = torch.full_like(mixed, 0.25)
    torch.testing.assert_close(
        inject_residual(block_output, residual, injection),
        (residual.unflatten(-1, (2, 8)) + 0.25).flatten(-2),
    )


@pytest.mark.unit
def test_ple_packed_output_matches_per_sample_execution():
    torch.manual_seed(11)
    config = tiny_config()
    module = Qwen4ExpPLE(config)
    tokens = torch.tensor([4, 5, 6, 20, 21, 22])
    hidden = torch.randn(6, 1, config.hyper_connection_width)
    packed = Qwen4ExpPackedLayout.from_cu_seqlens(6, torch.tensor([0, 3, 6]))

    packed_output = module(hidden, tokens, packed)
    separate_output = torch.cat(
        [
            module(
                hidden[start:end],
                tokens[start:end],
                Qwen4ExpPackedLayout.from_cu_seqlens(end - start, torch.tensor([0, end - start])),
            )
            for start, end in ((0, 3), (3, 6))
        ]
    )

    torch.testing.assert_close(packed_output, separate_output)


@pytest.mark.unit
def test_qsa_selection_stays_causal_and_inside_each_packed_sample():
    config = tiny_config()
    layout = Qwen4ExpPackedLayout.from_cu_seqlens(12, torch.tensor([0, 6, 12]))
    query = torch.randn(12, config.indexer_n_heads, config.indexer_head_dim)
    raw_keys = torch.randn(12, config.indexer_head_dim)
    cos, sin = build_rope_cos_sin(
        layout.positions,
        int(config.head_dim * config.partial_rotary_factor),
        config.rope_theta,
        query.dtype,
    )
    selected = qsa_select_token_indices(
        query,
        raw_keys,
        layout,
        torch.zeros(config.indexer_head_dim),
        config.rms_norm_eps,
        cos,
        sin,
        config.indexer_budget,
        config.indexer_compress_ratio,
    )

    for query_idx, indices in enumerate(selected):
        valid = indices[indices >= 0]
        sample_start = 0 if query_idx < 6 else 6
        assert bool(torch.all(valid >= sample_start))
        assert bool(torch.all(valid <= query_idx))
        visible_count = query_idx - sample_start + 1
        if visible_count % config.indexer_compress_ratio:
            assert query_idx in valid.tolist()


@pytest.mark.unit
def test_qsa_reference_freezes_indexer_and_backpropagates_lm_path():
    torch.manual_seed(7)
    config = replace(tiny_config(), indexer_budget=4, indexer_compress_ratio=2)
    module = Qwen4ExpSparseAttentionReference(config)
    hidden = torch.randn(6, 1, config.hidden_size, requires_grad=True)
    layout = Qwen4ExpPackedLayout.from_cu_seqlens(6, torch.tensor([0, 6]))

    output, selected = module(hidden, layout)
    output.square().mean().backward()

    assert output.shape == hidden.shape
    assert selected.shape == (6, 5)
    assert all(parameter.grad is None for parameter in module.indexer.parameters())
    assert module.q_proj.weight.grad is not None
    assert module.k_proj.weight.grad is not None
    assert module.v_proj.weight.grad is not None
    assert module.o_proj.weight.grad is not None


@pytest.mark.unit
def test_qsa_full_budget_path_matches_separate_packed_samples():
    torch.manual_seed(17)
    config = replace(tiny_config(), indexer_budget=8, indexer_compress_ratio=2)
    module = Qwen4ExpSparseAttentionReference(config)
    hidden = torch.randn(6, 1, config.hidden_size, requires_grad=True)
    packed = Qwen4ExpPackedLayout.from_cu_seqlens(6, torch.tensor([0, 3, 6]))

    packed_output, selected = module(hidden, packed)
    separate_output = torch.cat(
        [
            module(
                hidden[start:end],
                Qwen4ExpPackedLayout.from_cu_seqlens(end - start, torch.tensor([0, end - start])),
            )[0]
            for start, end in ((0, 3), (3, 6))
        ]
    )

    torch.testing.assert_close(packed_output, separate_output)
    assert selected.tolist() == [
        [0, -1, -1],
        [0, 1, -1],
        [0, 1, 2],
        [3, -1, -1],
        [3, 4, -1],
        [3, 4, 5],
    ]
