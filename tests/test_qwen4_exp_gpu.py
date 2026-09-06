"""Opt-in real FLA GDN + TE grouped MoE training test (four local CUDA GPUs).

QWEN4_EXP_RUN_GPU_TESTS=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
  torchrun --standalone --nproc-per-node=4 -m pytest tests/test_qwen4_exp_gpu.py -q -s

Uses a reduced architecture and synthetic policy data, without a public model
checkpoint or SGLang. Passing this test is not evidence of full-model RL closure.
"""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("QWEN4_EXP_RUN_GPU_TESTS") != "1", reason="opt-in four-GPU training test")
def test_real_gdn_moe_tp2_ep2_policy_step(monkeypatch):
    # Opting in makes absent dependencies/devices a failure, not a silent pass.
    assert torch.cuda.is_available(), "CUDA is required"
    assert int(os.environ.get("WORLD_SIZE", "0")) == 4, "launch with torchrun --nproc-per-node=4"
    import torch.distributed as dist
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from test_qwen4_exp_megatron_model import _mcore_config, _runtime_args
    from test_qwen4_exp_reference import tiny_config

    import slime_plugins.models.qwen4_exp.model as qwen4_model
    from slime.utils.ppo_utils import compute_policy_loss

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2, expert_model_parallel_size=2, expert_tensor_parallel_size=1
    )
    try:
        torch.manual_seed(31)
        model_parallel_cuda_manual_seed(31)
        p0 = replace(
            tiny_config(),
            hidden_size=128,
            head_dim=64,
            hc_lowrank=32,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            indexer_head_dim=32,
            indexer_budget=64,
            moe_intermediate_size=256,
            shared_expert_intermediate_size=256,
            vocab_size=256,
            eos_token_id=255,
            pad_token_id=255,
        )
        hf_text = SimpleNamespace(**p0.__dict__, hc_count=p0.hyper_connection_count, dtype=torch.bfloat16)
        # Only the public size contract and config IO are replaced. Every model
        # operator, TP/EP collective, and DDP gradient finalizer remains real.
        monkeypatch.setattr(qwen4_model.Qwen4ExpP0Config, "validate_public_release_contract", lambda _: None)
        monkeypatch.setattr(qwen4_model, "_load_hf_config", lambda _: SimpleNamespace(text_config=hf_text))
        args = _runtime_args(p0)
        args.tensor_model_parallel_size = 2
        args.expert_model_parallel_size = 2
        args.sequence_parallel = True
        args.transformer_impl = "transformer_engine"
        args.seq_length = args.max_position_embeddings = 64
        config = replace(
            _mcore_config(p0),
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            sequence_parallel=True,
            use_cpu_initialization=False,
            params_dtype=torch.bfloat16,
            bf16=True,
            transformer_impl="transformer_engine",
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            gradient_accumulation_fusion=False,
            moe_router_dtype="fp32",
        )
        model = qwen4_model.get_qwen4_exp_model_provider(args, config, None)().cuda()
        for parameter in model.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(parameter)
        wrapped = DistributedDataParallel(
            config, DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False), model
        )
        wrapped.broadcast_params()
        wrapped.zero_grad_buffer()
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        before = {name: p.detach().clone() for name, p in model.decoder.final_layernorm.named_parameters()}

        torch.manual_seed(41 + parallel_state.get_data_parallel_rank())
        tokens = torch.randint(0, 200, (1, 64), device="cuda")
        cu = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(
            cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=32, max_seqlen_kv=32, qkv_format="thd"
        )
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(
            wrapped(tokens, None, None, packed_seq_params=packed)
        )
        log_probs = logits.float().log_softmax(-1).gather(-1, tokens.roll(-1, -1).unsqueeze(-1)).squeeze(-1)
        advantages = torch.cat((torch.ones(32), -torch.ones(32))).cuda().unsqueeze(0)
        mask = (torch.arange(64, device="cuda") % 32 >= 23) & (torch.arange(64, device="cuda") % 32 < 31)
        pg_loss, _ = compute_policy_loss(log_probs.detach() - log_probs, advantages, 0.2, 0.28)
        loss = pg_loss[:, mask].mean()
        assert torch.isfinite(loss)
        loss.backward()
        finalize_model_grads([wrapped])
        gradients = {name: p.main_grad for name, p in model.named_parameters() if p.requires_grad}
        assert all(torch.isfinite(grad).all() for grad in gradients.values())
        for component in ("linear_attn", "mlp.experts", "self_attention.q_proj", "ple.norm_query", "final_layernorm"):
            assert any(grad.abs().sum() > 0 for name, grad in gradients.items() if component in name), component
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = parameter.main_grad.to(parameter.dtype)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        assert any(not torch.equal(p, before[name]) for name, p in model.decoder.final_layernorm.named_parameters())
        for parameter in model.decoder.final_layernorm.parameters():
            replicas = [torch.empty_like(parameter) for _ in range(2)]
            dist.all_gather(replicas, parameter, group=parallel_state.get_tensor_model_parallel_group())
            torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
        with torch.no_grad():
            updated = tensor_parallel.gather_from_tensor_model_parallel_region(
                wrapped(tokens, None, None, packed_seq_params=packed)
            )
        assert torch.isfinite(updated).all()
        assert not torch.equal(updated, logits)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
