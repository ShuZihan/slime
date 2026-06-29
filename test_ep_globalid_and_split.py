"""GPU-free verification for the Qwen3.5 INT4 EP global-expert-id plumbing.

Covers spec §5 step 3 (EP id mapping is disjoint & complete) and the logic half
of step 4 (offline converter split == runtime convert_to_hf split, bit-exact),
by importing and calling the REAL runtime function convert_qwen3_5_to_hf.
"""
import re
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/root/slime")
from slime.backends.megatron_utils.megatron_to_hf.qwen3_5 import convert_qwen3_5_to_hf

NUM_EXPERTS = 256
EP = 4
LOCAL = NUM_EXPERTS // EP  # 64 experts per EP rank
F = 512                    # moe-ffn-hidden-size
H = 2048                   # hidden-size

args = SimpleNamespace(num_experts=NUM_EXPERTS, kv_channels=None, hidden_size=H,
                       num_attention_heads=16, num_query_groups=2)


def common_py_offset(ep_rank):
    # mirrors common.py:_named_params_and_buffers_global
    #   expert_offset = ep_rank * args.num_experts // ep_size
    return ep_rank * NUM_EXPERTS // EP


def gate_ids(out):
    return sorted(int(re.search(r"experts\.(\d+)\.gate_proj", n).group(1)) for n, _ in out if "gate_proj" in n)


def down_ids(out):
    return sorted(int(re.search(r"experts\.(\d+)\.down_proj", n).group(1)) for n, _ in out)


# ---------------------------------------------------------------------------
# Test 1: EP global-id mapping is disjoint per rank and covers 0..255 exactly.
# ---------------------------------------------------------------------------
all_gate, all_down = [], []
for ep in range(EP):
    off = common_py_offset(ep)
    # Stage-1 name exactly as common.py emits it (gather-before rename, no slicing):
    name_fc1 = f"module.module.decoder.layers.0.mlp.experts.experts.linear_fc1.weight.__ep_offset{off}"
    param_fc1 = torch.zeros(LOCAL, 2 * F, H)
    out_fc1 = convert_qwen3_5_to_hf(args, name_fc1, param_fc1)
    g = gate_ids(out_fc1)
    assert g == list(range(off, off + LOCAL)), f"EP{ep} fc1 gate ids wrong: {g[:3]}..{g[-3:]} (offset {off})"

    name_fc2 = f"module.module.decoder.layers.0.mlp.experts.experts.linear_fc2.weight.__ep_offset{off}"
    param_fc2 = torch.zeros(LOCAL, H, F)
    out_fc2 = convert_qwen3_5_to_hf(args, name_fc2, param_fc2)
    d = down_ids(out_fc2)
    assert d == list(range(off, off + LOCAL)), f"EP{ep} fc2 down ids wrong (offset {off})"

    print(f"  EP{ep}: offset={off:3d} -> global ids [{off}, {off + LOCAL})  ✓")
    all_gate += g
    all_down += d

assert sorted(all_gate) == list(range(NUM_EXPERTS)), "gate global ids do NOT cover 0..255 uniquely"
assert sorted(all_down) == list(range(NUM_EXPERTS)), "down global ids do NOT cover 0..255 uniquely"
assert len(set(all_gate)) == NUM_EXPERTS, "gate global ids have duplicates across EP ranks"
print(f"PASS #1: 4 EP ranks map to disjoint ranges covering 0..{NUM_EXPERTS - 1} with no gaps/overlap")

# ---------------------------------------------------------------------------
# Test 2 (logic half of step 4): runtime convert_to_hf split is bit-identical
# to the offline converter split (both do param[i].chunk(2, dim=0) for gate|up,
# param[i] for down). Use real random data through the real function.
# ---------------------------------------------------------------------------
torch.manual_seed(1234)
fused_fc1 = torch.randn(LOCAL, 2 * F, H)
fused_fc2 = torch.randn(LOCAL, H, F)

out1 = convert_qwen3_5_to_hf(args, "module.module.decoder.layers.0.mlp.experts.experts.linear_fc1.weight.__ep_offset0", fused_fc1)
out2 = convert_qwen3_5_to_hf(args, "module.module.decoder.layers.0.mlp.experts.experts.linear_fc2.weight.__ep_offset0", fused_fc2)
hf = {n: t for n, t in list(out1) + list(out2)}

for i in range(LOCAL):
    # offline converter logic (tools/convert_hf_to_int4_direct.py):
    off_gate, off_up = fused_fc1[i].chunk(2, dim=0)
    off_down = fused_fc2[i]
    assert torch.equal(hf[f"model.language_model.layers.0.mlp.experts.{i}.gate_proj.weight"], off_gate)
    assert torch.equal(hf[f"model.language_model.layers.0.mlp.experts.{i}.up_proj.weight"], off_up)
    assert torch.equal(hf[f"model.language_model.layers.0.mlp.experts.{i}.down_proj.weight"], off_down)
print("PASS #2-logic: runtime convert_to_hf split == offline converter split, bit-exact (names + tensors)")

# ---------------------------------------------------------------------------
# Guard: a fused expert WITHOUT the __ep_offset suffix must still split at
# offset 0 (covers the offline path, which has no EP and reads global 0..255).
# ---------------------------------------------------------------------------
out_noff = convert_qwen3_5_to_hf(args, "module.module.decoder.layers.0.mlp.experts.experts.linear_fc1.weight", torch.zeros(LOCAL, 2 * F, H))
assert gate_ids(out_noff) == list(range(0, LOCAL)), "no-suffix fused split should start at global id 0"
print("PASS #3: no-suffix fused expert splits from global id 0 (offline-path compatible)")

print("\nALL CHECKS PASSED")
