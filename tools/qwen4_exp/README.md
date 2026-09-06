# Qwen4-Exp RL

在 slime 中运行 Qwen4-Exp / Qwen3.8-Flash-Next：HF 初始化 → SGLang rollout → GRPO 更新 → 在线同步 → 保存、导出与恢复。

当前 P0 为 BF16、纯文本、单条序列不超过 2048 token，PP=CP=ETP=1，PLE 大表和 QSA Indexer 冻结。下面的短序列示例使用 256 context、8 节点 × 8 张 NVIDIA GPU；实际容量仍需在目标机器确认。本地检查已通过，真实 GPU 和完整模型 RL 尚未验证。

## 开始运行

复用已有 `slime:latest`、完整模型目录和共享盘。head 能访问 GitHub，依赖源码放在共享盘供各节点使用。固定版本见 [versions.json](versions.json)。以下命令使用 Bash，按节点标注依次执行。

### 1. 拉取代码并填写配置（head 宿主机）

```bash
set -euo pipefail
export Q4_SHARED=/shared/qwen4-exp
export Q4_REPO_HOST="$Q4_SHARED/slime-qwen4-exp"
mkdir -p "$Q4_SHARED/ops"
git clone --branch qwen4-exp-rl --single-branch \
  https://github.com/ShuZihan/slime.git "$Q4_REPO_HOST"
git -C "$Q4_REPO_HOST" rev-parse HEAD | tee "$Q4_SHARED/ops/slime-commit.txt"
```

将下面的模型路径、head IP 和节点配置改成实际值：

```bash
cat > "$Q4_SHARED/ops.env" <<'ENV'
export Q4_SHARED=/shared/qwen4-exp
export Q4_REPO_HOST="$Q4_SHARED/slime-qwen4-exp"
export HF_CHECKPOINT=/shared/models/Qwen3.8-Flash-Next
export Q4_IMAGE=slime:latest
export Q4_HEAD_IP=REPLACE_WITH_HEAD_NODE_IP
export Q4_NODES=8
export Q4_GPUS_PER_NODE=8
export TRAIN_TP_SIZE=8
export TRAIN_EP_SIZE=8
ENV
source "$Q4_SHARED/ops.env"
```

已有 Git checkout 后续可用 `git pull --ff-only origin qwen4-exp-rl` 更新；先结束训练并保存本地修改，每次实验记录 `git rev-parse HEAD`。

### 2. 启动容器（每节点宿主机）

已有平台容器可以直接使用，保证 GPU 和以下路径挂载一致。`/dev/infiniband` 适用于有 RDMA 的机器；多机 DeepEP 需要相应互联支持。

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

### 3. 拉取并应用依赖补丁（head 容器，仅一次）

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

通过 `PYTHONPATH` 使用这些源码，复用镜像里的 Torch、CUDA、TE、FLA 和 DeepEP。使用新 `deps` 目录；网络中断后从失败命令继续，已成功的 clone/patch 无需重复。

### 4. 加载环境（每节点容器）

填写各节点自己的 IP。每次新开训练终端先 source 配置和依赖环境，再执行后续命令。

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

各节点使用相同 image ID 和依赖版本。Transformers 不匹配时需能访问配置的 pip 源；GitHub 可达不等于 PyPI 可达。若 router 检查失败，按 [Dockerfile](../../docker/Dockerfile) 中固定的 slime router wheel 补齐；算子/API 导入失败时先根据具体报错处理，源码补丁不会自动更新二进制库。

### 5. 启动 Ray（容器内）

head：

```bash
ray start --head --node-ip-address="$Q4_THIS_IP" \
  --port=6379 --dashboard-host=127.0.0.1 --dashboard-port=8265 \
  --num-gpus="$Q4_GPUS_PER_NODE" --disable-usage-stats
```

其余节点：

```bash
ray start --address="$Q4_HEAD_IP:6379" --node-ip-address="$Q4_THIS_IP" \
  --num-gpus="$Q4_GPUS_PER_NODE" --disable-usage-stats
```

head 设置提交地址并确认 GPU 节点就绪：

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

节点之间需开放 Ray、SGLang 和 NCCL/DeepEP 通信；代理旁路覆盖所有节点 IP。

### 6. 检查并运行两轮 RL（head 容器）

先运行本地回归：

```bash
cd /root/slime
export Q4_LOCAL_ID="local-$(date +%Y%m%d-%H%M%S)"
bash tools/qwen4_exp/validate.sh \
  --mode local --hf-checkpoint "$HF_CHECKPOINT" \
  --output-root "$Q4_SHARED/evidence" --run-id "$Q4_LOCAL_ID" \
  2>&1 | tee "$Q4_SHARED/evidence/$Q4_LOCAL_ID.console.log"
cat "$Q4_SHARED/evidence/$Q4_LOCAL_ID/result.json"
```

再在空闲的单节点上执行四 GPU 检查；它使用真实 GDN/MoE 的 tiny 配置：

```bash
export Q4_GPU_ID="gpu4-$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
QWEN4_EXP_RUN_GPU_TESTS=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --standalone --nproc-per-node=4 \
  -m pytest tests/test_qwen4_exp_gpu.py -q -s \
  2>&1 | tee "$Q4_SHARED/evidence/$Q4_GPU_ID.log"
```

通过后运行完整模型：

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

结果在 `$Q4_SHARED/evidence/$Q4_RUN_ID/result.json`，过程日志在 `e2e.log`；原始 checkpoint contract 要求 1658 个 tensor、131 个 shard。每次使用新的 run ID。

默认脚本执行两轮，每轮 8 prompts × 2 responses，训练和 rollout 分时复用 GPU。交替 0/1 奖励用于检查更新流程，不能用于判断回答质量。默认脚本没有配置保存，保存和恢复见下方。

## Loss、记录与可视化

**P0 先用 TensorBoard 看每步指标，保留样本用于复查。RL loss 不要求单调下降。** 当前每组人工奖励为 0/1，平均原始 reward 为 0.5；正负 advantage 可能使平均 policy loss 接近 0，但梯度仍然非零。默认两次更新只用于检查流程。

| 现有指标 | 判断重点 |
| --- | --- |
| `train/loss`、`train/pg_loss` | 无 NaN/Inf；当前 KL、entropy 系数为 0，两者应一致 |
| `train/grad_norm` | 有效 advantage 下应有梯度；排查持续为 0 或异常突增 |
| `train/train_rollout_logprob_abs_diff` | 同权重、同 token 下检查训推差异；`0.1` 只是流程门槛 |
| `train/pg_clipfrac`、`train/entropy_loss` | 关注大量裁剪、熵突然下降，并结合回答内容判断 |
| `train/lr-pg_*`、`train/global_batch_size` | 确认实际学习率和每步批大小符合配置 |
| `rollout/response_len/mean`、`rollout/truncated_ratio`、`rollout/repetition_frac` | 检查生成长度、截断和重复；默认回答上限仅 32 token |

### 开启记录

在第 6 步提交完整模型任务前，将 `scripts/run-qwen4-exp-p0-validation.sh` 复制为同目录下的 `run-qwen4-exp-p0-observe.sh`，在副本中做两处修改：

1. 在 `MISC_ARGS` 数组中加入：

   ```bash
   --use-tensorboard
   --save-debug-train-data "${QWEN4_EXP_VALIDATION_DIR}/train_data/{rollout_id}.pt"
   ```

2. 在构造 `RUNTIME_ENV_JSON` 的 Python `env` 字典中加入，确保 Ray 写入进程收到共享目录：

   ```python
   "TENSORBOARD_DIR": os.path.join(os.environ["QWEN4_EXP_VALIDATION_DIR"], "tensorboard"),
   ```

各节点镜像需能执行 `python3 -c 'from torch.utils.tensorboard import SummaryWriter'`。在第 6 步的 `validate.sh` 命令中增加 `--e2e-script /root/slime/scripts/run-qwen4-exp-p0-observe.sh`；它会设置本次 `QWEN4_EXP_VALIDATION_DIR`。原启动脚本不透传尾部参数，不能直接在 `bash scripts/…sh` 后追加训练 flag。

在安装了 TensorBoard、能访问共享目录的机器上查看本次实验：

```bash
tensorboard --logdir "$Q4_SHARED/evidence/$Q4_RUN_ID/tensorboard" \
  --host 127.0.0.1 --port 6006
```

浏览器打开 `http://127.0.0.1:6006`；远程运行时通过 SSH 转发 6006 端口，或将 event 文件复制到本地查看。训练曲线按 optimizer step，rollout 曲线按 rollout 轮次；先看未平滑曲线，避免漏掉尖峰。

- **每次更新：**现有日志记录上表训练指标；loss 来自本次更新使用的 forward，并非更新后的重算结果。
- **每轮采样：**`rollout_data/{rollout_id}.pt` 已默认开启；新增的 `train_data/{rollout_id}.pt` 保存 token、mask、advantage 和 logprob 等，可用 `rollout_position` 或 `sample_index` 与 rollout 样本关联。
- **每次实验：**使用独立 run ID，保留源码 commit、本地脚本改动、启动参数、模型/数据版本和随机种子，以及现有 console、`e2e.log`。

### 没有基线时如何验证

1. **计算正确：**从 dump 独立重算 GRPO loss，核对 token 对齐、温度、mask、裁剪区间和归约方式。默认先对每条回答的有效 token 求平均，再对样本求平均。
2. **更新正确：**固定同一批样本、旧策略 logprob 和 advantage，更新前后分别 forward；小步长下检查策略目标改善，并用 `lr=0` 对照参数不变。看 advantage 加权后的总体变化，不要求每条正奖励回答的概率都上升。
3. **效果改善：**换成与回答内容相关的奖励，以初始化 checkpoint 在固定独立评测集上的结果为基线，比较训练前后成功率及重复评测波动。

**待补能力：**现有 train dump 没有自动保存每步更新前后两套结果；固定 batch 更新对照、逐 token 误差的 p95/p99/max 和参数更新量需要额外采集。TensorBoard writer 的 `flush/close` 尚未接入完整收尾流程，需补每步 flush 或写入进程退出时的关闭，才能确保短 P0 的尾部记录落盘。曲线缺少末步数据时先核对 `e2e.log`，不能据此认定训练未执行。

工具选择：内网先用 [TensorBoard](https://docs.pytorch.org/docs/stable/tensorboard.html)；多实验协作可用已有 W&B 接口，offline 模式需后续同步到服务查看；已有统一实验平台时再考虑 MLflow。可视化和流程通过均不能代替精度对照。

## 保存、HF 导出与恢复

原始 checkpoint 约 360 GB，HF 导出与有状态 Adam checkpoint 还会增加大量磁盘、CPU 内存开销。保存时保留可读的原始 `HF_CHECKPOINT`，导出器需要补齐冻结权重。

<details>
<summary>展开保存和恢复命令</summary>

在 head 容器生成保存和恢复脚本副本；原启动脚本不透传尾部参数。保存副本改用有状态 Adam，恢复副本加载 optimizer 和 RNG：

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

保存两轮的 Megatron checkpoint 和 HF 模型：

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

产物在 `$QWEN4_EXP_SAVE_ROOT/megatron/` 和 `$QWEN4_EXP_SAVE_ROOT/hf/rollout-{0,1}/`。将模型输出与 evidence 分开，避免把模型打进日志压缩包。

从 iteration 1 恢复，继续执行 rollout 2、3，输出放入新目录：

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

恢复直接调用脚本，不套 `validate.sh`，因为其 HF 初始化审计不适用于 Megatron checkpoint 恢复。确认日志加载了 optimizer、RNG 和 `global_dataset_state_dict_1.pt`，并产生新一轮模型和 rollout 数据。保持原始 HF、训练拓扑和优化器配置一致。

</details>

<details>
<summary>展开 HF 导出模型重载检查</summary>

先结束 RL 任务、释放 GPU。每个节点容器填写 rank 和相同的导出目录，再启动 SGLang（下面仍为 8×8 示例）：

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

rank 0 服务 ready 后，在另一个 head 终端发送生成请求：

```bash
curl -fsS "http://$Q4_HEAD_IP:30000/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"What is 17 + 25?", "sampling_params":{"max_new_tokens":8,"temperature":0}}'
```

导出模型的 shard 数量可以不同。不要将导出目录传给针对原始模型固定 manifest 的检查；重载和生成成功也不等同于数值精度通过。

</details>

## 结果与限制

- `result.json.status=passed` 表示现有流程检查通过；失败先看 `stages.jsonl` 的第一个失败阶段及对应日志。
- 本地 72 项检查通过，四 GPU 测试尚未执行。单元测试、实际 GPU 流程和最终质量是不同层次的证据。
- 精度还需相同权重/输入下的独立前向与梯度、固定 batch 更新、训推逐 token 概率，以及保存恢复等价对照。默认聚合 logprob 阈值 `0.1` 不能代替这些检查。
- 机制、实现和后续范围见 [适配说明](../../docs/zh/developer_guide/qwen4-exp-rl-adaptation-analysis.md)。
