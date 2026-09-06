# Qwen4-Exp / slime P0 内网验证操作手册

版本：2026-09-06，修订 3（GitHub 直接拉取 + 现有 `slime:latest`）。代码与本手册位于 [ShuZihan/slime 的 qwen4-exp-rl 分支](https://github.com/ShuZihan/slime/tree/qwen4-exp-rl)，以本次实际 checkout 的 Git commit 为源码身份。下列示例经过本地检查；现有镜像兼容性、真实 checkpoint、GPU 与多机命令仍须在目标环境执行。

**主流程只有五步：克隆适配分支 → 用现有 `slime:latest` 启动容器 → GitHub 拉取固定依赖并应用补丁 → 启动 Ray → 运行验证。** 保存、导出、恢复见第 7–8 节；精度口径见第 9 节。正常准备流程无需传源码压缩包或重新构建、搬运镜像。

## 0. 执行范围和验收口径

本手册先建立可观察的 RL 闭环，再验证保存、导出、恢复，最后说明精度验收需要补齐的证据。命令以 **Linux x86_64、8 节点 × 8 张 NVIDIA GPU、节点间高速互联、同一共享目录**为配置示例。GPU 型号、显存、主机内存和驱动必须按实际填写；64 张卡不是容量已经验证的承诺。此路径使用 CUDA、NCCL、Transformer Engine、FLA、DeepEP，不是 Ascend/NPU 配方。

| 层次 | 本手册如何处理 | 可以得出的结论 |
|---|---|---|
| A：部署与流程 | 已有脚本，提供完整运行命令 | 两轮生成、评分、反传、更新、同步和再生成能够执行 |
| B：保存与交付 | 从原脚本生成保存/恢复副本，检查 HF 文件并重新加载 | checkpoint、HF 导出和恢复路径能够执行 |
| C：数值正确性 | 已有局部回归＋可手算策略检查；完整模型对照尚需补采集工具 | 必须有独立前向/梯度、固定 batch 更新、训推逐 token 对照才能验收 |
| D：学习效果 | 改用与回答内容有关的奖励，做固定评测与对照实验 | 模型在所选任务上出现可复现的学习收益 |

`validate.sh` 输出 `PASSED` 只表示其现有检查通过，不能自动把 C、D 标为完成。默认交替奖励与回答内容无关，默认 `logprob_threshold=0.1` 是流程门槛，均不是精度或质量的充分证据。

P0 边界：BF16、纯文本、每个样本总序列不超过 2048 token；当前闭环脚本实际使用 256。训练 PP=CP=ETP=1，使用 TP/SP、EP。PLE 大表和 QSA Indexer 冻结。Vision、MTP、FP8、长序列 QSA 不在本次训练验收范围内。

## 1. 只需准备这些

1. 从 `https://github.com/ShuZihan/slime.git` 拉取 **`qwen4-exp-rl` 分支**。本分支包含当前 P0 适配、正确性修复、回归测试和操作文档；普通 `main` 分支不包含这些改动。
2. 已有 `slime:latest` 镜像、完整 Qwen4-Exp checkpoint，以及可用的 GPU、互联和共享路径。模型目录包含 config、Tokenizer 和索引引用的全部真实 shard 文件。
3. head 容器能够拉取 GitHub。依赖源码只下载一次到共享盘，worker 直接使用，**不要求每个 worker 都访问 GitHub**。

复用镜像里的 Torch、CUDA、TE、FLA、DeepEP 和推理算子库。只覆盖 SGLang/Megatron 的 Python 源码；若 Transformers 不是 5.12.1，再安装这个固定版本。这一步可能需要可用的内网 pip 源，GitHub 可达不代表 PyPI 可达。算子/API 不兼容时以导入检查和四 GPU 检查定位，不自动升级整套 CUDA/Torch。

### 1.1 固定版本

以 `tools/qwen4_exp/versions.json` 为机器可读依据，不用 `latest` 替换这些身份。

| 项目 | 固定值 |
|---|---|
| SGLang | `78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385`，PR #36497 的固定源码 |
| SGLang overlay | `sglang-preserve-static-weight-reset.patch`、`sglang-qsa-short-extend.patch` |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087`，另需本仓库 Dockerfile 应用的 Megatron patches |
| 容器 Transformers | `5.12.1` |
| Transformers 参考源码 | `a8d5f2c845471633cf744c86be66bf888b8d6a37` |
| Torch / CUDA / TE / DeepEP | 复用 `slime:latest` 的实际版本；记录版本，并通过后续检查确认兼容性 |

各节点使用相同 `slime:latest` image ID、共享依赖源码和相同 Transformers 版本；同名 tag 本身不代表同一镜像。保留每节点实际版本记录。本流程通过 `PYTHONPATH` 选择源码，因此 SGLang/Megatron 的源码 commit 和实际 import 路径比 `pip show` 的旧安装元数据更直接。

### 1.2 模型与存储

原始公开 checkpoint contract 为 1658 个 tensor、131 个 shard 文件，索引 `metadata.total_size=359999963128` 字节，约 360 GB / 335.3 GiB。导出的 HF shard 数量和文件名可以不同，不能把“必须仍为 131 个文件”用于检查导出产物。

源权重约 360 GB；保存两个完整 HF 版本另约 720 GB；再加 Megatron checkpoint、Adam 状态、镜像、CPU offload、缓存和临时写入。按实际保留版本规划 TB 级磁盘，不能只留一个模型大小的空间。保存/恢复实验使用有状态 Adam，内存和 checkpoint 开销高于默认无状态 Adam。PLE host embedding 使用 pinned memory，CPU 内存不足也会失败；只查看 GPU 剩余显存不够。

## 2. 从 GitHub 拉取源码并填写路径

**执行位置：内网 head 宿主机。** 路径可整体替换，各节点容器必须看到相同的绝对路径。首次拉取：

```bash
set -euo pipefail
export Q4_SHARED=/shared/qwen4-exp
export Q4_REPO_HOST="$Q4_SHARED/slime-qwen4-exp"
mkdir -p "$Q4_SHARED/ops"
git clone --branch qwen4-exp-rl --single-branch \
  https://github.com/ShuZihan/slime.git "$Q4_REPO_HOST"
git -C "$Q4_REPO_HOST" rev-parse HEAD | tee "$Q4_SHARED/ops/slime-commit.txt"
```

如果这个目录已经是该分支的 Git checkout，后续更新执行下面一段，替代 clone。先结束使用该源码的训练进程，再更新；有本地改动时先保存到自己的提交或其他目录，检查失败时不要 reset/clean：

```bash
set -euo pipefail
export Q4_SHARED=/shared/qwen4-exp
export Q4_REPO_HOST="$Q4_SHARED/slime-qwen4-exp"
test -z "$(git -C "$Q4_REPO_HOST" status --porcelain)"
git -C "$Q4_REPO_HOST" fetch origin qwen4-exp-rl
git -C "$Q4_REPO_HOST" switch qwen4-exp-rl
git -C "$Q4_REPO_HOST" merge --ff-only origin/qwen4-exp-rl
mkdir -p "$Q4_SHARED/ops"
git -C "$Q4_REPO_HOST" rev-parse HEAD | tee "$Q4_SHARED/ops/slime-commit.txt"
```

旧交付包解出的目录没有 `.git`，应另选目录 clone。完整复现某次实验时使用该次记录的完整 commit，分支之后可能继续更新。第 7 节会生成脚本副本，保留它们的内容和 diff；运行中固定源码、依赖和参数，更新后重新执行相关检查并使用新的 run ID。

建议保存一份 `/shared/qwen4-exp/ops.env`，每次新开终端先 `source`。将下列 `REPLACE_*` 改成真实值：

```bash
export Q4_SHARED=/shared/qwen4-exp
export Q4_REPO_HOST="$Q4_SHARED/slime-qwen4-exp"
export HF_CHECKPOINT=/shared/models/Qwen3.8-Flash-Next
export Q4_IMAGE=slime:latest
export Q4_HEAD_IP=REPLACE_WITH_HEAD_NODE_IP
export Q4_NODES=8
export Q4_GPUS_PER_NODE=8
export TRAIN_TP_SIZE=8
export TRAIN_EP_SIZE=8
```

`Q4_THIS_IP` 每节点不同，在第 3 节各节点终端中单独设置，不写入这份共享 `ops.env`。保存与日志目录用共享盘；Ray 临时目录用各节点本地盘。

## 3. 用现有镜像启动容器并准备源码

### 3.1 每节点启动容器

已有可用的平台容器可直接进入容器，确认相同挂载后跳到 3.2。其余情况在每台宿主机执行：

```bash
source /shared/qwen4-exp/ops.env
docker image inspect --format '{{.Id}}' "$Q4_IMAGE"
docker run -d --name qwen4-exp-p0 \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device=/dev/infiniband \
  --mount "type=bind,src=$Q4_SHARED,dst=$Q4_SHARED" \
  --mount "type=bind,src=$Q4_REPO_HOST,dst=/root/slime" \
  --mount "type=bind,src=$HF_CHECKPOINT,dst=$HF_CHECKPOINT,readonly" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e WANDB_MODE=disabled \
  -e RAY_USAGE_STATS_ENABLED=0 \
  -e PYTHONPATH=/root/slime \
  -e HF_CHECKPOINT="$HF_CHECKPOINT" \
  -w /root/slime "$Q4_IMAGE" bash -lc 'exec tail -f /dev/null'
docker exec -it qwen4-exp-p0 bash
```

`/dev/infiniband` 必须真实存在，且平台允许映射。没有 RDMA 的节点不能直接照抄该参数；当前多机 DeepEP 配方的通信可用性需另行确认。不要用临时禁用 RDMA/NCCL 检查来掩盖互联问题。

### 3.2 head 容器：GitHub 拉取源码并打补丁（只做一次）

以下命令只在 head 容器运行一次，结果放在共享盘。使用新目录，保留镜像自带的仓库和已有实验。网络失败时修复后从失败的 Git 命令继续，已成功的 clone/patch 无需重复；不要在 worker 同时执行此段。

```bash
set -euo pipefail
source /shared/qwen4-exp/ops.env
cd /root/slime
export Q4_DEPS="$Q4_SHARED/deps"
test ! -e "$Q4_DEPS"
mkdir -p "$Q4_DEPS"

git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang.git "$Q4_DEPS/sglang"
git -C "$Q4_DEPS/sglang" fetch --no-tags origin refs/pull/36497/head
git -C "$Q4_DEPS/sglang" checkout --detach 78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385
for patch in sglang-preserve-static-weight-reset.patch sglang-qsa-short-extend.patch; do
  git -C "$Q4_DEPS/sglang" apply --check "/root/slime/docker/patch/qwen4-exp/$patch"
  git -C "$Q4_DEPS/sglang" apply "/root/slime/docker/patch/qwen4-exp/$patch"
done

git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/Megatron-LM.git "$Q4_DEPS/Megatron-LM"
git -C "$Q4_DEPS/Megatron-LM" checkout --detach 1dcf0dafa884ad52ffb243625717a3471643e087
for patch in megatron.patch megatron-sglang-aligned.patch; do
  git -C "$Q4_DEPS/Megatron-LM" apply --check "/root/slime/docker/patch/latest/$patch"
  git -C "$Q4_DEPS/Megatron-LM" apply "/root/slime/docker/patch/latest/$patch"
done

cat > "$Q4_DEPS/env.sh" <<'ENV'
export Q4_REPO=/root/slime
export MEGATRON_ROOT="$Q4_SHARED/deps/Megatron-LM"
export Q4_SGLANG_ROOT="$Q4_SHARED/deps/sglang"
export PYTHONPATH="$Q4_REPO:$MEGATRON_ROOT:$Q4_SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
ENV
```

该步骤只拉取固定源码、应用四个补丁并写入环境文件，保留镜像里的算子库。当前 P0 用 HTTP、纯文本 rollout；扩展到 gRPC/多模态等路径时，不能据此假设对应 Rust 扩展也已准备好。

### 3.3 每节点容器：加载环境并检查依赖

每次新开训练终端先执行前面的 `source` 和环境设置。每个节点单独填写本节点 IP，head 的 `Q4_THIS_IP` 必须等于 `Q4_HEAD_IP`。

```bash
set -euo pipefail
source /shared/qwen4-exp/ops.env
source "$Q4_SHARED/deps/env.sh"
export Q4_THIS_IP=REPLACE_WITH_THIS_NODE_IP
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export RAY_USAGE_STATS_ENABLED=0
export NO_PROXY="127.0.0.1,localhost,$Q4_HEAD_IP,$Q4_THIS_IP${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
cd /root/slime
mkdir -p "$Q4_SHARED/evidence" "$Q4_SHARED/saved" "$Q4_SHARED/ops"

# 版本已匹配时不调用 pip；仅缺少/不匹配时安装固定 Transformers。
if ! python3 -c 'from importlib.metadata import version; assert version("transformers") == "5.12.1"'; then
  python3 -m pip install --no-deps 'transformers==5.12.1'
fi
python3 - <<'PY'
import os, subprocess
from pathlib import Path
import torch, transformers, megatron.core as megatron, sglang, slime
assert torch.cuda.is_available()
assert transformers.__version__ == '5.12.1'
for module, root, commit in (
    (sglang, os.environ['Q4_SGLANG_ROOT'], '78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385'),
    (megatron, os.environ['MEGATRON_ROOT'], '1dcf0dafa884ad52ffb243625717a3471643e087'),
):
    Path(module.__file__).resolve().relative_to(Path(root).resolve())
    assert subprocess.check_output(['git','-C',root,'rev-parse','HEAD'],text=True).strip() == commit
    print(module.__name__, module.__file__)
assert slime.__file__.startswith('/root/slime/')
import transformer_engine.pytorch, fla, deep_ep, sglang_router
import sglang.srt.models.qwen4_exp
from sglang.srt.entrypoints.http_server import launch_server
from slime_plugins.models.qwen4_exp.model import Qwen4ExpTransformerLayer
assert 'slime' in sglang_router.__version__, 'need the slime-compatible sglang_router build'
print('torch:',torch.__version__,'CUDA:',torch.version.cuda,'Transformers:',transformers.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i,p.name,p.total_memory)
print('source paths, pins and imports passed')
PY
python3 tools/qwen4_exp/check_sglang_patch.py --sglang-root "$Q4_SGLANG_ROOT"
python3 -m pip freeze > "$Q4_SHARED/ops/pip-$Q4_THIS_IP.txt"
nvidia-smi -q > "$Q4_SHARED/ops/gpu-$Q4_THIS_IP.txt"
```

如果 pip 不可达，为这一条安装命令配置已有内网源（`PIP_INDEX_URL`），或准备对应 wheel；无需重建镜像。如果导入失败，先保留 traceback 和上述版本记录，根据具体缺失包/ABI 修复。普通 `latest` 的组件组合未知，源码下载成功本身不能证明二进制兼容，后续四 GPU 检查仍需通过。

如果**只有 router 检查不通过**，可按仓库 Dockerfile 使用的固定构建安装后重跑检查：

```bash
python3 -m pip install --no-deps \
  https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl
```

该 wheel 限 Linux x86_64；同版本号的其他 router 构建可能使 pip 判为 already installed，此时仅对上述 wheel 加 `--force-reinstall`。其他算子不做批量升级。已有镜像完全满足检查时，这一步不执行。

## 4. 检查模型并启动 Ray

### 4.1 检查模型完整性

在 head 容器运行以下完整性检查；第 6 节还会逐节点检查索引引用的 shard 是否可见。索引大小与文件存在只检查结构，不检查每个大文件的全部数据；传输时仍应核对供应方 checksum。

```bash
cd /root/slime
python3 - <<'PY'
import json, os
from pathlib import Path
from slime.utils.hf_config import load_hf_config
from transformers import AutoTokenizer
p = Path(os.environ['HF_CHECKPOINT'])
index = json.loads((p / 'model.safetensors.index.json').read_text())
files = sorted(set(index['weight_map'].values()))
assert len(index['weight_map']) == 1658
assert len(files) == 131
assert index['metadata']['total_size'] == 359999963128
assert all((p / f).is_file() and (p / f).stat().st_size > 0 for f in files)
cfg = load_hf_config(p)
tokenizer = AutoTokenizer.from_pretrained(str(p), local_files_only=True, trust_remote_code=True)
ids = tokenizer.apply_chat_template([{'role':'user','content':'What is 17 + 25?'}],
                                    tokenize=True, add_generation_prompt=True)
assert ids
print(type(cfg).__name__, type(tokenizer).__name__, 'prompt tokens:', len(ids))
PY
```

`trust_remote_code` 仅用于已经带入并审核的本地模型内容；离线开关不补全缺失的 tokenizer/code 文件。

### 4.2 启动 Ray

确保是本次专用集群。head/worker 都在上述容器中启动，并已 `source "$Q4_SHARED/deps/env.sh"`。Ray 启动后修改依赖不会更新现存进程的导入状态，因此先完成第 3 节再启动。

head：

```bash
ray start --head --node-ip-address="$Q4_THIS_IP" \
  --port=6379 --dashboard-host=127.0.0.1 --dashboard-port=8265 \
  --num-gpus="$Q4_GPUS_PER_NODE" --disable-usage-stats
```

其余 7 节点分别执行（填写本节点 `Q4_THIS_IP`）：

```bash
ray start --address="$Q4_HEAD_IP:6379" --node-ip-address="$Q4_THIS_IP" \
  --num-gpus="$Q4_GPUS_PER_NODE" --disable-usage-stats
```

head 容器：

```bash
export RAY_ADDRESS="$Q4_HEAD_IP:6379"
export RAY_DASHBOARD_ADDRESS=http://127.0.0.1:8265
ray status --address="$RAY_ADDRESS"
python3 - <<'PY'
import os, ray
ray.init(address=os.environ['RAY_ADDRESS'])
nodes = [n for n in ray.nodes() if n.get('Alive') and n.get('Resources',{}).get('GPU',0)>0]
print([(n['NodeManagerAddress'], n['Resources'].get('GPU')) for n in nodes])
assert len(nodes) == int(os.environ['Q4_NODES'])
assert all(int(n['Resources']['GPU']) == int(os.environ['Q4_GPUS_PER_NODE']) for n in nodes)
ray.shutdown()
PY
```

head dashboard 只绑定本机，因为操作器在 head 提交。节点之间还需要 Ray worker/object manager、NCCL/DeepEP、SGLang 分布式初始化等通信；仅开放 6379/8265 不充分。用内网既有的训练网络策略，不把 Ray 控制端口暴露到公网。代理旁路列表需覆盖全部训练节点；示例只列 head/本机，按实际补全。[Ray 集群 CLI](https://docs.ray.io/en/latest/cluster/cli.html)。

## 5. 先运行低成本检查

### 5.1 metadata + 本地回归

head 容器：

```bash
cd /root/slime
export Q4_LOCAL_ID="local-$(date +%Y%m%d-%H%M%S)"
bash tools/qwen4_exp/validate.sh \
  --mode local --hf-checkpoint "$HF_CHECKPOINT" \
  --output-root "$Q4_SHARED/evidence" --run-id "$Q4_LOCAL_ID" \
  2>&1 | tee "$Q4_SHARED/evidence/$Q4_LOCAL_ID.console.log"
cat "$Q4_SHARED/evidence/$Q4_LOCAL_ID/result.json"
```

要求 `result.json.status=passed`，检查 `unit.log` 中 skip 原因。`--mode local` 不检查完整多机运行环境，不运行四 GPU 测试，也不运行 RL。单元测试的 pass 数量随文件版本变化，验收依赖具体测试与证据，不能只对照历史“71 passed”。

### 5.2 四 GPU 真实 GDN / MoE 训练检查

在没有其他 GPU 任务的一台节点运行。先完成这个检查，再提交完整模型；不要和 RL 同时占用同一批卡。

```bash
export Q4_GPU_ID="gpu4-$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
QWEN4_EXP_RUN_GPU_TESTS=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --standalone --nproc-per-node=4 \
  -m pytest tests/test_qwen4_exp_gpu.py -q -s \
  2>&1 | tee "$Q4_SHARED/evidence/$Q4_GPU_ID.log"
```

此测试是 tiny 配置的 TP2/EP2 真实 FLA GDN、TE grouped MoE、DDP 归约与单步更新，检查 frozen 参数不变、关键梯度非零、final mixer 副本一致。它不加载完整 checkpoint，不启动 SGLang，也没有独立整模数值参考。显式启用后缺少设备/依赖应当失败，不能把 skip 当通过。

### 5.3 可手算的策略方向检查

以下直接调用实际 `compute_policy_loss`，并以独立手算梯度检查无 clipping 的初始点。它验证策略损失的符号和一个 mask 案例，不代替模型 backward 或完整 GRPO 数据流验证。

```bash
python3 - <<'PY'
import torch
from slime.utils.ppo_utils import compute_policy_loss
for a in (1.0, -1.0, 0.0):
    z = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    lp = z.log_softmax(0)[0]
    old = lp.detach()
    loss, _ = compute_policy_loss(old-lp, torch.tensor(a), 0.2, 0.28)
    loss.backward()
    expected = torch.tensor([-0.5*a, 0.5*a], dtype=torch.float64)
    torch.testing.assert_close(z.grad, expected, rtol=1e-10, atol=1e-10)
    before = z.softmax(0)[0].item()
    after = (z.detach()-0.01*z.grad).softmax(0)[0].item()
    assert (after>before if a>0 else after<before if a<0 else after==before)
    print('advantage', a, 'gradient', z.grad.tolist(), 'p(a)', before, '->', after)
z = torch.zeros((2,2), dtype=torch.float64, requires_grad=True)
lp = z.log_softmax(-1)[:,0]
loss, _ = compute_policy_loss(lp.detach()-lp, torch.ones(2), 0.2, 0.28)
(loss * torch.tensor([1.,0.])).sum().backward()
assert z.grad[1].abs().sum() == 0
print('policy sign and direct-loss mask checks passed')
PY
```

真实模型的 prompt/padding 对后续输出可能有间接影响，不能将此独立 logits 检查扩展成“mask=0 位置对应的所有模型参数都不应有梯度”。

## 6. 原样执行两轮 RL 闭环

### 6.1 固定配方

| 参数 | 当前脚本值/含义 |
|---|---|
| actor | 8 节点 × 8 GPU；TP8、SP 开启、EP8、PP1/CP1/ETP1 |
| rollout | 一个跨 64 GPU 的 engine；DP=EP=64；DP attention / DP LM head |
| GPU 使用 | `--colocate`，训练与 rollout 分时复用同一批 GPU，伴随 offload/onload |
| 样本 | 每轮 8 prompts × 2 responses；global batch 16；共 2 轮 |
| 长度 | 最大 prompt 192、最大 context 256、最大 response 32 |
| 奖励 | 按组顺序交替 0/1，用于产生 advantage，不用于评估回答质量 |
| 优化器 | Stateless Adam，lr=1e-4，无 weight decay；用于放大可观察更新 |
| 同步 | full + NCCL；排除冻结 PLE/Indexer 等静态参数 |
| 保存 | 原脚本未配置 `--save` / `--save-hf`，并设置 `--no-save-optim` |

改变 GPU 数量时，`--expected-*` 会改变 actor/rollout 总规模，但训练 TP/EP 仍需显式设置：`world % TP == 0`，`(world/TP) % EP == 0`，`512 % EP == 0`。合法整除关系不是显存或 DeepEP 支持的保证。首轮按已整理配方执行，不把并行、kernel、精度和奖励同时改动。

### 6.2 提交

head 容器：

```bash
cd /root/slime
export TRAIN_TP_SIZE=8 TRAIN_EP_SIZE=8
export RAY_ADDRESS="$Q4_HEAD_IP:6379"
export RAY_DASHBOARD_ADDRESS=http://127.0.0.1:8265
export Q4_RUN_ID="p0-loop-$(date +%Y%m%d-%H%M%S)"
bash tools/qwen4_exp/validate.sh \
  --mode intranet --hf-checkpoint "$HF_CHECKPOINT" \
  --ray-address "$RAY_ADDRESS" \
  --expected-nodes "$Q4_NODES" \
  --expected-gpus "$((Q4_NODES * Q4_GPUS_PER_NODE))" \
  --output-root "$Q4_SHARED/evidence" --run-id "$Q4_RUN_ID" \
  --logprob-threshold 0.1 \
  2>&1 | tee "$Q4_SHARED/evidence/$Q4_RUN_ID.console.log"
```

必须使用 Bash 的 `set -o pipefail`，否则 `tee` 成功会掩盖训练命令失败。run ID 用无点号、未使用过的名字；现有验收器要求目录不存在，归档名用 `Path.with_suffix` 生成，带点号的 run ID 可能发生意外重名。

确认未改源码且第 5.1 节已通过时，可以在后续重复运行中加 `--skip-unit`。不要首轮跳过。`validate.sh` 没有 `--preflight-only`、`--save-hf` 或任意训练参数透传功能；不要向它添加不存在的参数。

### 6.3 阶段与通过标准

| 错误码 | 实际检查 | 必看产物 |
|---|---|---|
| Q4E-100 | 本地 CUDA/固定版本、SGLang API/patch；Ray 节点、GPU、共享 marker、关键源码 hash | `preflight.json`、`ray-preflight.json`、`stages.jsonl` |
| Q4E-200 | 原始公开模型 index/生命周期 contract | `manifest-summary.json`、`parameter-lifecycle.jsonl` |
| Q4E-300 | 选定本地测试文件通过 | `unit.log` |
| Q4E-400 | 两轮 RL 命令退出成功 | `e2e.log`、`rollout_data/0.pt`、`1.pt` |
| Q4E-410 | 所有 actor rank 的 HF 加载审计、关键 tensor 抽样对齐、PLE 分片 | `checkpoint/rank-*.jsonl`、`checkpoint-alignment-summary.json` |
| Q4E-500 | 抽样 trainable 指纹变化，静态/派生状态保持 | `model/rank-*.jsonl`、`model-transition-summary.json` |
| Q4E-510 | engine rank checksum 覆盖、trainable 变化、PLE/Indexer 不变 | `rollout/engine-checksums.jsonl`、`engine-sync-summary.json` |
| Q4E-600 | 聚合 logprob 差异、非零有限梯度、非空生成、奖励差异、权重版本改变 | `rl-closure-summary.json` |
| Q4E-900 | 日志与审计目录归档 | 同级 `.tar.gz` 和 `.tar.gz.sha256` |

门禁中的指纹/加载审计有抽样范围；Q4E-510 的“权重变了”也不等于“所有更新后的 tensor 都与训练端正确对应”。Q4E-600 的奖励差异不能作为严谨的逐组 GRPO 正确性证明。精度验收见第 9 节。

```bash
export Q4_RUN_DIR="$Q4_SHARED/evidence/$Q4_RUN_ID"
cat "$Q4_RUN_DIR/result.json"
python3 - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['Q4_RUN_DIR'])
for line in (p/'stages.jsonl').read_text().splitlines():
    x = json.loads(line)
    if x.get('event') in ('stage_passed','stage_failed'):
        print(x.get('code'), x.get('stage'), x.get('event'), x.get('error',''))
assert json.loads((p/'result.json').read_text())['status'] == 'passed'
PY
```

一个终端等待提交结束，另一个查看 `tail -f "$Q4_RUN_DIR/e2e.log"`。不要仅看到 `ray job submit` 的 submission ID 就认定任务成功。

## 7. 保存 checkpoint、HF 导出与恢复

### 7.1 生成独立保存/恢复副本

原始 `scripts/run-qwen4-exp-p0-validation.sh` 不转发命令行尾部参数，不能用 `bash 原脚本 --save ...` 开启保存。下面在同一 `scripts` 目录生成两个副本，保证脚本相对路径仍正确；不修改原脚本。

保存副本移除 `--no-save-optim` 和 `--use-stateless-adam`，改为持久化 Adam 状态，每轮保存；恢复副本另移除 `--no-load-optim`，从完整 Megatron checkpoint root 恢复。**两者必须配套使用**，不能从默认无状态 Adam 运行中恢复并声称 optimizer history 已恢复。

```bash
cd /root/slime
python3 - <<'PY'
from pathlib import Path
src = Path('scripts/run-qwen4-exp-p0-validation.sh')
s = src.read_text()
for needle in ('  --no-save-optim\n', '  --use-stateless-adam\n'):
    assert s.count(needle) == 1, needle
    s = s.replace(needle, '')
needle = 'CKPT_ARGS=(\n'
assert s.count(needle) == 1
s = s.replace(needle, needle +
    '  --save "${QWEN4_EXP_SAVE_ROOT:?}/megatron"\n'
    '  --save-interval 1\n'
    '  --save-hf "${QWEN4_EXP_SAVE_ROOT:?}/hf/rollout-{rollout_id}"\n')
save = src.with_name('run-qwen4-exp-p0-save-validation.sh')
resume = src.with_name('run-qwen4-exp-p0-resume-validation.sh')
assert not save.exists() and not resume.exists(), 'use fresh filenames; inspect existing copies first'
save.write_text(s)
save.chmod(0o755)
needle = '  --load "${HF_CHECKPOINT}"\n'
assert s.count(needle) == 1
r = s.replace(needle, '  --load "${QWEN4_EXP_RESUME_FROM:?}"\n')
assert r.count('  --no-load-optim\n') == 1
r = r.replace('  --no-load-optim\n', '')
assert r.count('  --num-rollout 2\n') == 1
r = r.replace('  --num-rollout 2\n', '  --num-rollout "${QWEN4_EXP_RESUME_TOTAL_ROLLOUTS:?}"\n')
resume.write_text(r)
resume.chmod(0o755)
print(save, resume)
PY
bash -n scripts/run-qwen4-exp-p0-save-validation.sh
bash -n scripts/run-qwen4-exp-p0-resume-validation.sh
diff -u scripts/run-qwen4-exp-p0-validation.sh scripts/run-qwen4-exp-p0-save-validation.sh || test "$?" -eq 1
```

保留 HF 源目录作为 `--hf-checkpoint`。它提供配置、Tokenizer 和导出时补齐的静态权重；恢复训练的 `--load` 则指向 Megatron checkpoint，两者用途不同。

### 7.2 保存与导出运行

```bash
export Q4_SAVE_ID="p0-save-$(date +%Y%m%d-%H%M%S)"
export QWEN4_EXP_SAVE_ROOT="$Q4_SHARED/saved/$Q4_SAVE_ID"
test ! -e "$QWEN4_EXP_SAVE_ROOT"
mkdir -p "$QWEN4_EXP_SAVE_ROOT"
bash tools/qwen4_exp/validate.sh \
  --mode intranet --hf-checkpoint "$HF_CHECKPOINT" \
  --ray-address "$RAY_ADDRESS" \
  --expected-nodes "$Q4_NODES" --expected-gpus "$((Q4_NODES * Q4_GPUS_PER_NODE))" \
  --e2e-script /root/slime/scripts/run-qwen4-exp-p0-save-validation.sh \
  --output-root "$Q4_SHARED/evidence" --run-id "$Q4_SAVE_ID" \
  --skip-unit \
  2>&1 | tee "$Q4_SHARED/evidence/$Q4_SAVE_ID.console.log"
```

`saved` 与 `evidence` 分开，避免验收器把数百 GB 的 HF 模型和 optimizer checkpoint 再打入日志压缩包。该运行增加了保存成本和 Adam 状态，若容量不足，先按实际参数量与 offload 方式分析；不要删除 optimizer 保存后仍把实验标成完整恢复验收。

预期产物：

```text
saved/<save-id>/
  megatron/
    latest_checkpointed_iteration.txt
    iter_0000000/ ...
    iter_0000001/ ...
    rollout/global_dataset_state_dict_0.pt
    rollout/global_dataset_state_dict_1.pt
  hf/
    rollout-0/config.json + tokenizer assets + safetensors index/shards
    rollout-1/config.json + tokenizer assets + safetensors index/shards
```

具体 Megatron shard 格式由固定依赖和实际配置决定，不按文件名猜内容。`{rollout_id}` 是源码实际使用的命名占位符。

### 7.3 检查 HF 导出结构

```bash
export Q4_EXPORTED_HF="$QWEN4_EXP_SAVE_ROOT/hf/rollout-1"
python3 - <<'PY'
import json, os
from pathlib import Path
from safetensors import safe_open
src, dst = Path(os.environ['HF_CHECKPOINT']), Path(os.environ['Q4_EXPORTED_HF'])
def index(p):
    return json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
a, b = index(src), index(dst)
assert a.keys() == b.keys(), (sorted(a.keys()-b.keys()), sorted(b.keys()-a.keys()))
for name in a:
    with safe_open(src/a[name], framework='pt', device='cpu') as f:
        sa, da = tuple(f.get_slice(name).get_shape()), f.get_slice(name).get_dtype()
    with safe_open(dst/b[name], framework='pt', device='cpu') as f:
        sb, db = tuple(f.get_slice(name).get_shape()), f.get_slice(name).get_dtype()
    assert (sa,da)==(sb,db), (name,sa,sb,da,db)
assert (dst/'config.json').is_file()
print('HF names/shapes/dtypes passed:', len(b), 'tensors;', len(set(b.values())), 'shards')
PY
```

这一步通过 [safetensors slice API](https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs) 检查 names/shapes/dtypes，不证明 tensor 值正确，也不证明模型可生成。下一节必须实际重载。导出 shard 数量不同会改变固定 manifest digest，**不要把导出的 HF 目录作为原始公开 checkpoint 输入 `validate.sh` 的 Q4E-200 contract**。

### 7.4 恢复训练

先保存之前的根路径，再为恢复运行使用新输出目录：

```bash
export QWEN4_EXP_RESUME_FROM="$QWEN4_EXP_SAVE_ROOT/megatron"
cat "$QWEN4_EXP_RESUME_FROM/latest_checkpointed_iteration.txt"
test "$(tr -d '[:space:]' < "$QWEN4_EXP_RESUME_FROM/latest_checkpointed_iteration.txt")" = 1
test -f "$QWEN4_EXP_RESUME_FROM/rollout/global_dataset_state_dict_1.pt"
export Q4_RESUME_ID="p0-resume-$(date +%Y%m%d-%H%M%S)"
export QWEN4_EXP_SAVE_ROOT="$Q4_SHARED/saved/$Q4_RESUME_ID"
export QWEN4_EXP_VALIDATION_DIR="$Q4_SHARED/evidence/$Q4_RESUME_ID"
export QWEN4_EXP_RESUME_TOTAL_ROLLOUTS=4
export ACTOR_NUM_NODES="$Q4_NODES" ACTOR_NUM_GPUS_PER_NODE="$Q4_GPUS_PER_NODE"
export ROLLOUT_NUM_GPUS="$((Q4_NODES * Q4_GPUS_PER_NODE))"
export ROLLOUT_NUM_GPUS_PER_ENGINE="$ROLLOUT_NUM_GPUS"
export ROLLOUT_DP_SIZE="$ROLLOUT_NUM_GPUS" ROLLOUT_EP_SIZE="$ROLLOUT_NUM_GPUS"
test ! -e "$QWEN4_EXP_VALIDATION_DIR"
test ! -e "$QWEN4_EXP_SAVE_ROOT"
mkdir -p "$QWEN4_EXP_VALIDATION_DIR" "$QWEN4_EXP_SAVE_ROOT"
bash scripts/run-qwen4-exp-p0-resume-validation.sh \
  2>&1 | tee "$QWEN4_EXP_VALIDATION_DIR/e2e.log"
```

这里显式要求上一轮 checkpoint iteration 为 1，自动从 rollout 2 开始，`num_rollout=4` 意味着执行 rollout 2、3，而不是再执行 4 轮。如果断言失败，先确认上一节是否完整执行、是否选对 checkpoint 目录。不要加 `--finetune`、`--no-load-optim`、`--no-load-rng` 或强制 `--start-rollout-id 0`。保持同一原始 HF 模型、训练拓扑、优化器、数据集与随机设置。

恢复测试**直接调用脚本副本**，不套原始 `validate.sh`：原操作器的 Q4E-410 要求 HF 初始化 audit，而 Megatron checkpoint 恢复本来就不走 HF tensor 加载。这里没有自动 `result.json`，按下列条件人工记录：

- 日志明确从 Megatron checkpoint 加载，未重新从 HF 初始化 actor，未报告跳过 optimizer/RNG 加载。
- 读取了对应 `global_dataset_state_dict_1.pt`，不是只恢复参数却把数据游标重置。
- 新 `rollout_data/2.pt`、`3.pt`、模型指纹、同步 checksum 与新保存产物存在；过程退出成功。
- 要验收“恢复等价”，还需与连续运行到同一步的固定 batch 参数更新对照。在线随机采样后的文本相同/不同都不足以代替该对照，详见第 9 节。

## 8. 导出模型重新加载与固定序列概率采集

### 8.1 独立启动 SGLang

先确认 RL job 和其 GPU engine 已退出，释放同一批 GPU。以下是导出重载 smoke，不与 RL 并行运行；使用同一 8 节点拓扑，在**每个节点容器**设置 rank 0–7 并启动。`Q4_EXPORTED_HF` 指向第 7 节完整导出目录，所有节点路径一致。

```bash
export Q4_NODE_RANK=REPLACE_WITH_0_TO_7
export Q4_EXPORTED_HF=/shared/qwen4-exp/saved/REPLACE_SAVE_ID/hf/rollout-1
python3 -m sglang.launch_server \
  --model-path "$Q4_EXPORTED_HF" --dtype bfloat16 \
  --host "$Q4_THIS_IP" --port 30000 \
  --nnodes "$Q4_NODES" --node-rank "$Q4_NODE_RANK" \
  --dist-init-addr "$Q4_HEAD_IP:29500" \
  --tp-size 64 --dp-size 64 --ep-size 64 \
  --enable-dp-attention --enable-dp-lm-head \
  --moe-dp-size 1 --moe-dense-tp-size 1 \
  --moe-a2a-backend deepep --deepep-mode auto \
  --mem-fraction-static 0.70 --page-size 64 \
  --kv-cache-dtype bfloat16 --context-length 256 \
  --max-prefill-tokens 512 --chunked-prefill-size 512 --max-running-requests 16 \
  --language-model-only --ple-offload-embedding --linear-attn-backend triton \
  --disable-radix-cache --disable-overlap-schedule --disable-prefill-cuda-graph \
  --enable-deterministic-inference --watchdog-timeout 7200 --dist-timeout 1800 \
  2>&1 | tee "$Q4_SHARED/evidence/export-reload-node-$Q4_NODE_RANK.log"
```

上述 `tp/dp/ep=64` 对应本手册的 8×8 示例，缩放时一起核对。端口 29500/30000 须未占用且节点互通。等待 rank 0 服务 ready 后，在另一个 head 终端测试；完成后在这些专用终端停止本次服务，勿用全局进程名清理。

### 8.2 保存固定 token IDs，采集逐 token logprob

固定 token 文件生成一次后复用，不能在两次对照中各自重新套 chat template。

```bash
export Q4_TOKEN_FILE="$Q4_SHARED/evidence/fixed-token-ids.json"
python3 - <<'PY'
import json, os
from pathlib import Path
from slime.utils.hf_config import load_hf_config
from transformers import AutoTokenizer
root = os.environ['HF_CHECKPOINT']
load_hf_config(root)
t = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=True)
ids = t.encode('The sum of seventeen and twenty-five is forty-two.', add_special_tokens=False)
assert 2 <= len(ids) <= 256
p = Path(os.environ['Q4_TOKEN_FILE'])
assert not p.exists(), 'reuse the existing fixture; do not silently overwrite'
p.write_text(json.dumps(ids)+'\n')
print('fixed tokens:', len(ids))
PY
```

下面使用已核对的 `/generate` 输入概率接口，不产生新 token。只用于已加载的目标模型；URL 是内网 head 服务。

```bash
export Q4_ENDPOINT="http://$Q4_HEAD_IP:30000"
export Q4_LOGPROB_OUT="$Q4_SHARED/evidence/export-reload-logprobs.json"
python3 - <<'PY'
import json, math, os, urllib.request
from pathlib import Path
ids = json.loads(Path(os.environ['Q4_TOKEN_FILE']).read_text())
body = {'input_ids':ids, 'sampling_params':{'max_new_tokens':0,'temperature':1.0},
        'return_logprob':True, 'logprob_start_len':0}
req = urllib.request.Request(os.environ['Q4_ENDPOINT']+'/generate',
    data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=600) as r:
    out = json.load(r)
rows = out['meta_info']['input_token_logprobs']
assert len(rows) == len(ids), (len(rows),len(ids))
assert [x[1] for x in rows] == ids
assert all(x[0] is not None and math.isfinite(float(x[0])) for x in rows[1:])
p = Path(os.environ['Q4_LOGPROB_OUT'])
assert not p.exists(), p
p.write_text(json.dumps({'input_ids':ids,'logprobs':[x[0] for x in rows],
                         'endpoint':os.environ['Q4_ENDPOINT']},ensure_ascii=False)+'\n')
print('teacher-forced tokens checked:',len(ids)-1,'output:',p)
PY
```

第一个 token 没有前文概率，比较从第二个 token 开始。若接口返回格式不符，保留原响应、核对固定依赖，不删除 token ID 对齐断言来凑结果。再执行解码 smoke，保存原始响应：

```bash
python3 - <<'PY'
import json, math, os, urllib.request
from pathlib import Path
ids = json.loads(Path(os.environ['Q4_TOKEN_FILE']).read_text())
body = {'input_ids':ids, 'sampling_params':{'max_new_tokens':4,'temperature':0.0},
        'return_logprob':True}
req = urllib.request.Request(os.environ['Q4_ENDPOINT']+'/generate',
    data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=600) as r:
    out = json.load(r)
p = Path(os.environ['Q4_SHARED'])/'evidence'/'export-reload-generate.json'
assert not p.exists(), p
p.write_text(json.dumps(out,ensure_ascii=False)+'\n')
rows = out['meta_info']['output_token_logprobs']
assert rows and all(x[0] is not None and math.isfinite(float(x[0])) for x in rows)
print('decode smoke passed; tokens:',len(rows),'response:',out.get('text',''))
PY
```

这能执行“导出可重载并计算概率”。完整数值验证还需要另一端相同权重/输入的概率文件。**原始模型与训练后导出模型本来就应该不同，不能把二者概率接近当作训练正确性目标。**

## 9. 从零到一的精度验收

### 9.1 现成检查和缺失工具的界线

当前有局部数学/packed 回归、TP mixer 梯度回归、checkpoint mapping/export 测试，以及可选真实 GPU 活动检查。当前没有一条已经实现的命令能自动完成“独立 Qwen4 整模参考 → Megatron → 在线 SGLang → 导出重载”的全部逐 token/梯度对照。

`reference.py` 的实现被训练模型直接复用，因此与自身对比不能充当独立模型语义证据。Transformers 5.12.1 的配置注册 fallback 也不等于拥有完整 HF `AutoModel` 实现。若参考源码/环境尚不能执行，明确将相应验收标为 `NOT RUN / blocked by missing reference runner`，不要虚构 `--precision-check` 一类命令。

### 9.2 需要形成的六份报告

| 报告 | 具体操作与固定条件 | 必需证据 |
|---|---|---|
| 初始化前向 | 相同源 checkpoint、token IDs、positions/masks；先模块，再完整层，再整模 | 首次偏差位置、逐层 hidden/logits、逐 token logprob；输入/权重身份 |
| 独立梯度 | tiny FP64/FP32 数学实现；对关键输入/参数做数值差分，避开路由切换和其他不可微边界 | forward 与 gradient 误差、差分步长扫描；真实 kernel 对独立参考的结果 |
| 并行等价 | tiny TP1/EP1 对 TP2/SP/EP2，固定全局 batch、loss 分母和路由；再验证目标拓扑 | 还原全局布局后的梯度和一次 optimizer 更新；不能只看 grad norm |
| 固定 batch 更新 | 固定 tokens、old/ref logprob、advantage、mask、optimizer 初态；与独立公式对照 | loss 分项、选定参数梯度/增量、正负/零 advantage 与 mask 案例 |
| 在线同步等价 | 保存同一步 live actor；比较在线同步 engine 与从同一步完整 HF 重新启动的 engine | 全部适用 tensor 的映射/值检查＋固定序列 logprob；不能只比较版本号 |
| 恢复等价 | 连续训练 N+1 步与保存 N 步后恢复再训练一步；使用同一固定 batch 和 RNG/optimizer/scheduler 状态 | 参数增量、loss、optimizer/scheduler 状态、数据游标一致性 |

前四项的完整 runner、在线 engine 同步前后概率采集 hook、固定 batch 恢复对照尚需补充。可先按本手册完成部署和功能 smoke，但这些报告未完成时，结论只能写“流程跑通、精度待验收”。数值差分检查 backward 与 forward 的一致性，仍需独立公式核对 forward 本身。

### 9.3 固定数据集与误差门槛

固定输入至少包含：短序列、接近 2048 上界、不同长度 packed/分开执行、EOS 与 PLE 历史重置、padding、batch 顺序变化。默认 256 context 配方不覆盖 2048 上界；做边界实验时同步修改模型 `--seq-length`、rollout max context、SGLang context 等所有限制，并单独记录该变体，不能只改一个长度参数。

先在同一实现测重复运行波动，再分别测 FP32→BF16、普通算子→融合算子、非分布式→分布式引入的误差；据此制定并冻结门槛。不要在看到候选实现结果后放宽门槛。采集最大绝对误差、P95/P99、超限比例与异常 token/layer；梯度同时看归一化误差与方向，近零梯度不能只看相对误差。

`0.1` 是默认**聚合** logprob 差异门槛，不能读成“每个 token 都小于 0.1”。同一 token 若 logprob 相差 0.1，其概率比约偏离 10.5%；对 RL ratio/clip 的影响要单独量化。不同 backend/归约顺序不保证浮点逐 bit 一致。[PyTorch 数值精度](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)、[梯度检查](https://docs.pytorch.org/docs/2.14/notes/gradcheck.html)。

### 9.4 学习趋势实验

默认交替奖励与内容无关、lr=1e-4 为观察参数变化而设，不要直接将它作为正式 RL 配方。精度检查后，另建与回答内容关联且容易评价的任务，例如限定答案格式的简单算术或受控 token 选择；先确保初始模型既有成功又有失败，奖励不是全常数。

固定训练问题、独立评测集、采样预算与评测参数。对比初始模型/不更新对照和训练模型；用多个随机种子重复，报告 reward/成功率、长度、entropy、KL 和有效 token 数。只看训练 reward 上升不能排除长度偏好、奖励漏洞或数据泄漏。正式学习率与 KL 等参数需单独校准，记录为新的实验配置。

## 10. 失败定位与证据收集

先找 `stages.jsonl` 中第一个失败阶段，再看该阶段原始日志；最后的 Ray actor died/NCCL timeout 可能只是传播后的报错。

| 现象 | 优先检查 |
|---|---|
| CUDA/TE/FLA/DeepEP import 或 `no kernel image` | 现有镜像与目标 GPU capability、驱动/CUDA/扩展 ABI 是否匹配；源码覆盖不更新二进制算子 |
| SGLang/Megatron version mismatch | 实际 import 路径、image ID、固定源码及 overlay；不要改期望 commit 来绕过 |
| Ray GPU 数量或源码 hash mismatch | 是否在每个相同镜像容器启动 Ray、挂载路径/PYTHONPATH是否一致、是否混入其他集群节点 |
| checkpoint shard missing/manifest mismatch | 符号链接、复制完整性、模型版本是否就是固定公开 contract |
| GPU OOM / 主机被杀 | GPU 峰值与 host/pinned memory 分开查；colocate offload、PLE、Adam 状态和并发加载峰值 |
| 训练停在 collective / engine 初始化 | 所有节点首个错误、节点 IP、代理、RDMA设备与接口、NCCL/DeepEP配置；别直接扩大超时掩盖错误 |
| Q4E-410 | 加载 audit 是否覆盖全部 ranks，首次 mismatch 的 tensor、专家编号、TP PLE offset |
| Q4E-500/510 | 静态参数是否进入更新；实际抽样 trainable 是否变化；engine ranks 是否完整 |
| Q4E-600 | 是否有两轮数字 dump，权重版本是否改变，logprob 配对/temperature/mask，梯度与奖励详情 |
| 保存目录为空 | 是否运行保存副本；`--save-interval`、`--save`、`--save-hf` 是否实际进入训练命令 |
| 恢复后从零开始/缺 optimizer | 是否指向 checkpoint root；是否残留 no-load-optim/finetune/stateless；数据游标是否恢复 |
| 导出重载失败 | 权重名/shape、全局专家重组、源静态权重、Tokenizer；不要只数 shard 文件 |

head 可用 `ray job list --address="$RAY_DASHBOARD_ADDRESS"` 找到本次 submission ID，再用 `ray job logs <SUBMISSION_ID> --address=...` 或 `ray job stop <SUBMISSION_ID> --address=...`。只操作本次任务 ID，不按进程名全局清理。源码/配置修正后使用新的 run ID，保留失败证据。每次实验另存当时的 `slime-commit.txt`，不要只依赖会被后续更新覆盖的共享记录。

打包前收集：

- `ops/slime-commit.txt`、Git 工作区 diff/status、所有生成脚本副本、完整命令与 env 中非秘密配置。
- 每节点 GPU/驱动/内存、image ID、pip freeze、实际 import 路径。
- `result.json`、`stages.jsonl`、`e2e.log`、各 summary、checkpoint/model/engine audit、rollout dumps。
- 本次 Ray submission ID、Ray head 与 worker 的相关日志，以及第一处 GPU/NCCL/DeepEP traceback。
- 第 7 节保存/恢复运行的输出目录清单；第 8、9 节固定输入、概率文件和比较报告。

原验收器自动归档的目录不包含所有外部 Ray 日志；失败时需要补收。Ray 常见日志位于启动节点的 `/tmp/ray/session_latest/logs`，以实际 Ray session 路径为准，及时保留本次相关日志。不要把完整模型权重混进小型故障包；日志和 rollout 可能包含训练数据，按内网的数据流程处理。

```bash
cd "$Q4_SHARED/evidence"
sha256sum -c "$Q4_RUN_ID.tar.gz.sha256"
```

恢复运行是直接脚本调用，需自行归档对应 evidence 目录。已有自动归档在创建时尚未记录最后的 archive completion 事件，判断最终结果同时保留磁盘上的 `result.json` 和校验文件。

## 11. 最终验收记录模板

| 项目 | 状态：PASS / FAIL / NOT RUN | 证据路径与备注 |
|---|---|---|
| 当前源码/镜像/模型身份一致 | | |
| 全节点预检与本地测试 | | |
| 四 GPU 真实 GDN/MoE | | |
| 完整模型两轮 RL + 在线同步 | | |
| 有状态 Adam checkpoint 保存 | | |
| 完整 HF 导出结构与重载 | | |
| checkpoint 恢复及数据游标 | | |
| 独立前向与梯度数值对照 | | |
| TP/SP/EP 与固定 batch 更新等价 | | |
| 在线同步与同一步 HF 重载概率对照 | | |
| 连续训练与恢复后下一步等价 | | |
| 与回答内容相关的学习趋势/固定评测 | | |

报告结论分别写“功能闭环通过”“数值对照通过”“任务收益通过”，并注明 checkpoint、Git commit 与本地 diff、GPU 拓扑、精度与序列范围。任何未执行项保持 NOT RUN。

## 源码定位

本手册核对的主要入口：`tools/qwen4_exp/{versions.json,build_environment.sh,validate.py,check_sglang_patch.py}`、`scripts/run-qwen4-exp-p0-validation.sh`、`scripts/models/qwen4-exp.sh`、`train.py`、`slime/backends/megatron_utils/{actor.py,model.py,checkpoint.py,hf_checkpoint_saver.py,qwen4_exp_hf_export.py}`、`slime_plugins/models/qwen4_exp/{validation.py,validation_reward.py,lifecycle.py}`、`tests/test_qwen4_exp_gpu.py`。独立 SGLang 启动和 logprob 接口对照固定 SGLang 源码的 `server_args.py`、`io_struct.py` 和已有 logprob tests。

本次简化还修正了验收器的 Megatron 版本定位：探测实际 `megatron.core` 文件，避免把 namespace package 的空 `megatron.__file__` 传给 `Path`；训练和模型算法未因此改变。
