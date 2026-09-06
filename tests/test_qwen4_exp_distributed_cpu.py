"""Real CPU/Gloo TP regressions; these do not execute GDN or the MoE forward."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreSavePlanner,
    TorchDistSaveShardedStrategy,
    _replace_state_dict_keys_with_sharded_keys,
    mcore_to_pyt_state_dict,
)
from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads
from test_qwen4_exp_megatron_model import FakeGatedDeltaNet, _mcore_config, _runtime_args
from test_qwen4_exp_reference import tiny_config

import slime_plugins.models.qwen4_exp.model as qwen4_model


class _DiscardedNorm(torch.nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _CpuSaveStrategy(TorchDistSaveShardedStrategy):
    """Keep MCore's mapping/planner; use PyTorch's synchronous CPU file writer.

    The production async writer requires CUDA staging and forks IO workers,
    neither of which is portable to this CPU/macOS regression environment.
    """

    def save(self, state, path):
        import torch.distributed.checkpoint as dcp

        state, _, _ = _replace_state_dict_keys_with_sharded_keys(state, self.keep_only_main_replica)
        dcp.save_state_dict(
            mcore_to_pyt_state_dict(state, False),
            dcp.FileSystemWriter(path),
            planner=MCoreSavePlanner(flatten_state_dict=False),
        )


def _tp_worker(rank, rendezvous, checkpoint, case):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        torch.manual_seed(31)
        p0 = tiny_config()
        hf_text = SimpleNamespace(**p0.__dict__, hc_count=p0.hyper_connection_count)
        args, config = _runtime_args(p0), _mcore_config(p0)
        args.tensor_model_parallel_size = config.tensor_model_parallel_size = 2
        args.sequence_parallel = config.sequence_parallel = True
        get_block_spec = qwen4_model.get_gpt_decoder_block_spec

        def cpu_block_spec(*args, **kwargs):
            spec = get_block_spec(*args, **kwargs)
            # This temporary GPT norm is replaced by the real Qwen final mixer.
            spec.layer_norm = _DiscardedNorm
            return spec

        with (
            patch.object(qwen4_model, "get_gpt_decoder_block_spec", cpu_block_spec),
            patch.object(qwen4_model, "Qwen3_5GatedDeltaNet", FakeGatedDeltaNet),
            patch.object(qwen4_model.Qwen4ExpP0Config, "validate_public_release_contract", lambda _: None),
            patch.object(qwen4_model, "_load_hf_config", lambda _: SimpleNamespace(text_config=hf_text)),
        ):
            model = qwen4_model.get_qwen4_exp_model_provider(args, config, None)()
        if case == "mixer":
            _check_mixer_gradients(model, config, rank, p0)
        else:
            _check_ple_checkpoint(model, rank, checkpoint)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _check_mixer_gradients(model, config, rank, p0):
    mixer = model.decoder.final_layernorm
    reference = copy.deepcopy(mixer)
    torch.manual_seed(17)
    inputs = torch.randn(8, 1, p0.hyper_connection_width)
    target = torch.randn(8, 1, p0.hidden_size)
    # Local losses use the full-batch denominator, as sequence-parallel shards do.
    ((mixer(inputs.chunk(2)[rank]) - target.chunk(2)[rank]) ** 2).sum().div(target.numel()).backward()
    ((reference(inputs) - target) ** 2).mean().backward()
    mixer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    for param in mixer.parameters():
        param.main_grad = param.grad.clone()
    _allreduce_non_tensor_model_parallel_grads([mixer], config, parallel_state.get_tensor_model_parallel_group())
    for param, expected in zip(mixer.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(param.main_grad, expected.grad, rtol=1e-5, atol=1e-7)
        # Also check the optimizer-visible update, not only the metadata marker.
        torch.testing.assert_close(param - 0.1 * param.main_grad, expected - 0.1 * expected.grad)
    for layer in model.decoder.layers:
        for module in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
            assert not any(getattr(p, "sequence_parallel", False) for p in module.parameters())


def _check_ple_checkpoint(model, rank, checkpoint):
    # Exercise the actual GPT -> block -> custom layer -> nested embedding path.
    layer = model.decoder.layers[1]
    table = layer.ple.ple_embedding.ngram_embedding
    with torch.no_grad():
        rows = torch.arange(table.vocab_start_index, table.vocab_end_index).float()
        table.weight.copy_(rows[:, None].expand_as(table.weight))
    metadata = {"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)}
    state = model.sharded_state_dict(metadata=metadata)
    key = "decoder.layers.1.ple.ple_embedding.ngram_embedding.weight"
    shard = state[key]
    assert shard.global_shape == (table.num_embeddings, table.embedding_dim)
    assert shard.global_offset == (table.vocab_start_index, 0)
    assert shard.axis_fragmentations == (2, 1)
    assert shard.key == key  # Heterogeneous layers must not share a stacked checkpoint key.
    ple_state = {name: value for name, value in state.items() if name.startswith("decoder.layers.1.ple.")}
    expected = {name: value.data.clone() for name, value in ple_state.items()}
    if rank == 0:
        Path(checkpoint).mkdir()
    dist.barrier()
    dist_checkpointing.save(ple_state, checkpoint, sharded_strategy=_CpuSaveStrategy("torch_dist", 1))
    with torch.no_grad():
        for value in ple_state.values():
            value.data.zero_()
    loaded = dist_checkpointing.load(ple_state, checkpoint)
    for name, tensor in loaded.items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


@pytest.mark.integration
@pytest.mark.parametrize("case", ["mixer", "checkpoint"])
def test_real_tp2_cpu_regressions(tmp_path, case):
    mp.spawn(_tp_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path / "checkpoint"), case), nprocs=2)
