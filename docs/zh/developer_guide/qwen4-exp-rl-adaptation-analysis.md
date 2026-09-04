# Qwen4-Exp 在 slime 中的 RL 适配设计与交付

> 术语约定：公开 checkpoint 的仓库名为 `Qwen3.8-Flash-Next`，配置字段 `model_type=qwen4_exp`、Transformers 类名和 SGLang 实现均使用 `Qwen4Exp`。本文统一称模型及架构为 **Qwen4-Exp**；公开 checkpoint 路径、外部链接和配置字段保留上游名称。

## 交付结论

当前分支已经完成 Qwen4-Exp 纯文本 RL P0 的代码实现，训练链路为：

`HF safetensors → Megatron actor → generate/reward/backward/step → Megatron-to-HF 在线转换 → SGLang rollout`

P0 固定以下范围：

- text-only；vision、MTP、speculative decoding 和量化 rollout 关闭；
- 51B PLE embedding table 与 12 层 QSA Indexer 冻结，启动时分别加载到 Megatron 和 SGLang；
- PLE 其余 projection、norm、gate、conv 参数参与训练和在线同步；
- TP+EP，`CP=1`、`PP=1`、`ETP=1`；
- 验证上下文为 256 tokens，落在 QSA `indexer_budget=2048` 内，训练侧执行等价的 full-selection causal SDPA；
- 本地 CPU/tiny-model 测试覆盖结构、数学语义、梯度、checkpoint 映射、packed sequence、同步过滤和验收器；完整 checkpoint、多机 GPU 及真实 SGLang 在线闭环由内网验收命令执行。

实现入口：

| 入口 | 作用 |
|---|---|
| [`slime_plugins/models/qwen4_exp/model.py`](../../../slime_plugins/models/qwen4_exp/model.py) | Megatron 模型 provider 与完整 decoder layer |
| [`slime_plugins/models/qwen4_exp/reference.py`](../../../slime_plugins/models/qwen4_exp/reference.py) | GR、PLE、QSA 和 packed 语义实现 |
| [`slime_plugins/models/qwen4_exp/config.py`](../../../slime_plugins/models/qwen4_exp/config.py) | 公开 checkpoint 架构契约 |
| [`slime_plugins/models/qwen4_exp/lifecycle.py`](../../../slime_plugins/models/qwen4_exp/lifecycle.py) | 参数生命周期与在线同步集合 |
| [`hf_to_megatron/qwen4_exp.py`](../../../slime/backends/megatron_utils/hf_to_megatron/qwen4_exp.py) | HF→Megatron 映射、PLE 流式分片加载 |
| [`megatron_to_hf/qwen4_exp.py`](../../../slime/backends/megatron_utils/megatron_to_hf/qwen4_exp.py) | Megatron→SGLang 权重名和布局转换 |
| [`scripts/models/qwen4-exp.sh`](../../../scripts/models/qwen4-exp.sh) | 公开模型的 Megatron 参数 |
| [`tools/qwen4_exp/validate.sh`](../../../tools/qwen4_exp/validate.sh) | 本地与内网分阶段验收 |

## 固定版本与模型身份

所有运行时版本集中记录在 [`tools/qwen4_exp/versions.json`](../../../tools/qwen4_exp/versions.json)：

| 依赖 | 固定值 |
|---|---|
| slime 基线 | `4c1ab40203952b3dcc8582b653f3a83f2c6e8128` |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087` |
| SGLang | PR [#36497](https://github.com/sgl-project/sglang/pull/36497) 的 `78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385` |
| Transformers 参考源码 | `a8d5f2c845471633cf744c86be66bf888b8d6a37` |
| 容器 Transformers | `5.12.1` |
| 默认 CUDA | `13.0.3` |

Transformers `5.12.1` wheel 未包含 Qwen4-Exp 原生注册。[`slime/utils/hf_config.py`](../../../slime/utils/hf_config.py) 在 SGLang 可用时注册其 `Qwen4ExpConfig`；CPU 工具环境使用字段兼容的 config-only fallback。Tokenizer 绑定公开 checkpoint 共用的 Qwen2 tokenizer 类。Transformers 固定源码承担数学参考来源，SGLang 固定提交承担 rollout 实现来源。

公开 checkpoint 的 text contract：

| 字段 | 值 |
|---|---:|
| 主模型 / 激活参数 | 125B / 约 6B |
| hidden size / layer | 2560 / 48 |
| layer pattern | `3 × GDN + 1 × QSA`，共 36 层 GDN、12 层 QSA |
| mHC | 4 路 residual stream，low-rank 320，stream width 10240 |
| QSA | 24 query heads、2 KV heads、head dim 256 |
| RoPE | partial factor 0.25，theta 10,000,000 |
| QSA Indexer | 4 query heads、1 KV head、head dim 128 |
| QSA selection | token budget 2048、compress ratio 4、block Top-K 512 |
| PLE | 第 2 层，一基编号；embed dim 2560；trigram；每个 n-gram 8 heads |
| PLE vocabulary | base 20,000,000；divisor 128；128 个 source shards |
| MoE | 512 routed experts、Top-10、Top-K 概率归一化、routed/shared intermediate size 640 |
| vocab / EOS / effective PAD | 248320 / 248044 / 248044 |

完整 checkpoint index 固定为 1,658 个 tensor、131 个文件、metadata `total_size=359999963128`。参数生命周期 manifest 的 SHA-256 为 `00ae2f76f403da35de3f9cb05514c7720affa1bac907e7a90f0f4ae2f3c9e56d`。

## 核心不变量

同一批 token 和同一 actor 版本满足：

`logits_megatron(x, θ_step, θ_static) ≈ logits_sglang(x, θ_step, θ_static)`

`θ_step` 在 optimizer step 后更新并传给 rollout engine；`θ_static` 在进程启动时加载，并在整个 RL run 内保持指纹不变。Packed execution 对每个样本产生与单独 execution 相同的有效 token 结果。TP/EP 只改变参数和 token ownership，GR、PLE、QSA 与 MoE 数学语义保持一致。

一次有效 RL 闭环还满足：

1. 第 0 次 rollout 记录权重版本 `v`，并产生非空 response 与组内 reward 方差；
2. Megatron 用该批数据计算 rollout/train logprob 差异、有限且非零的 gradient norm；
3. optimizer step 后至少一个抽样 trainable tensor 指纹变化，所有抽样 static tensor 指纹保持不变；
4. SGLang 同步前后至少一个 trainable tensor checksum 变化，PLE/Indexer checksum 保持不变；
5. 第 1 次 rollout 记录新权重版本 `v+1` 并成功生成。

## Megatron 训练图

### mHC residual state

Token embedding 首先产生 `H=2560` 的 hidden state。第一个 decoder layer 将其复制成 4 路 residual stream，后续层保持 `4H=10240`：

`R_l ∈ ℝ[...,4H] → GR_read(R_l) ∈ ℝ[...,H] → Block → GR_write(Block,R_l) ∈ ℝ[...,4H]`

Attention 和 MoE 各拥有一套 `Qwen4ExpGatedResidual`。Read path 对 `4H` 做以 `H` 为 group size 的 zero-centered RMSNorm，经 `4H→320→4H` 低秩门控后混合四路输入；write path 生成四个 injection gate，将 `H` 宽 block output 写回每路 residual stream。Decoder 末端使用同结构的全局 mixer 将 `4H` 收回 `H`，随后进入 TP vocabulary output layer。Critic 接入点位于该最终 mixer 之后。

自定义 GR、GDN、QSA 与 PLE 小参数在 P0 的 TP ranks 上复制；sequence-parallel token 在进入自定义 block 前 gather，进入原生 Megatron MoE 前 scatter。该路径优先建立确定的训练语义。Embedding、LM head、PLE table 与 MoE 使用 Megatron 原生并行 ownership。

### GDN

36 个 linear-attention layer 复用 slime 的 `Qwen3_5GatedDeltaNet` 与 varlen `cu_seqlens` 接口。Qwen4-Exp checkpoint 的 output gate 为 `sigmoid`；共享实现读取 `output_gate_type`，缺省模型继续读取 `hidden_act`。这项修改保持 Qwen3.5 配置行为，并使 Qwen4-Exp 与参考实现一致。

### QSA

QSA 将历史 key 每 4 tokens 压缩成一个 block key，公开配置的选择预算为 2048 tokens。P0 的训练序列固定为 256 tokens，因此选择集合覆盖完整 causal prefix，训练侧逐个 packed sample 调用 causal SDPA。Q/K/V/O 路径参与 autograd；Indexer 三组 checkpoint 参数保留在模型状态中并冻结，用于维持 Megatron 与 SGLang 的启动权重一致性。

训练实现只接受 `max_seqlen ≤ 2048`。模型 provider 同时约束 `seq_length`，运行时再次检查每个 packed sample。超出预算的输入立即返回明确错误。本分支不携带长上下文 Top-K oracle、selected-index 输出和 query-chunk recompute 路径。

### PLE

PLE 在第 2 个 decoder layer、attention 前注入。输入 token 生成 bigram/trigram hash ids，经约 51B 参数的 n-gram table 查表，再执行 key/value projection、query-key gate 和 dilation=3 的 depthwise short convolution。

PLE table 的 BF16 容量约 95 GiB。Megatron 使用 `VocabParallelEmbedding` 按行切到 TP ranks；HF loader 依次打开 128 个 source tensors，仅物化与本 rank 行区间相交的 source shard，并检查覆盖区间连续、完整。三个 hash metadata buffer 从 config 生成，随后与 checkpoint tensor 做逐值校验。

P0 冻结 table，projection、norm 与 convolution 继续训练。SGLang 启用 `ple_offload_embedding`，其启动权重与 Megatron 来自同一 checkpoint。

### MoE

每层包含 512 个 routed experts、Top-10 router 和一个 shared expert。训练侧使用 Megatron grouped GEMM、AllToAll dispatcher、`EP=8`、`ETP=1`。HF grouped expert tensor 按 global expert id 做 slice；在线反向映射保持 global id，随后由 EP gather 还原 SGLang 所需权重。Checkpoint 的 `norm_topk_prob=true`；Megatron 使用 Top-K logits 上的 softmax，得到同样的归一化权重。

## Packed sequence 语义

[`slime/backends/megatron_utils/data.py`](../../../slime/backends/megatron_utils/data.py) 把多个样本拼成单行 token stream，并建立 `cu_seqlens`。Qwen4-Exp 模型据此生成：

- 每个样本从 0 开始的 `positions`；
- 可直接遍历的样本起止位置；
- PLE n-gram 与 convolution 的重置边界；
- QSA 候选集合与 causal mask 的样本边界。

公开 checkpoint 的 `pad_token_id=null`，训练参数解析时回退到 `eos_token_id=248044`。为满足 TP padding 倍数而添加的尾部 token 作为一个独立 synthetic packed sample 写入 `cu_seqlens`，其 loss mask 全零。PLE 还会在样本内部的 EOS 后重置词法 n-gram history。

P0 以 `CP=1` 保持完整样本局部可见，并以 `PP=1` 保持 `4H` residual state 位于同一 pipeline stage。

## Checkpoint 与参数生命周期

manifest 对 1,658 个 source tensor 逐项分类，遇到未知名称立即失败：

| 生命周期 | 数量 | 训练图 | optimizer | 每步在线同步 |
|---|---:|---:|---:|---:|
| `trainable_sync` | 1127 | 是 | 是 | 是 |
| `static_shared` | 164 | 是 | 否 | 否 |
| `derived_buffer` | 3 | 是 | 否 | 否 |
| `disabled` | 364 | 否 | 否 | 否 |

`static_shared` 包含 128 个 PLE table shards 与 12 层 × 3 个 QSA Indexer tensors。`disabled` 覆盖 vision tower 和 1 层 MTP。`derived_buffer` 覆盖 PLE multipliers、head vocabulary sizes 和 offsets。

HF→Megatron 映射覆盖 embedding、LM head、全局 mixer、每层 GR、GDN、QSA、PLE 小参数、router、grouped routed experts 与 shared expert。QSA 的 `q_proj` 同时携带 query 与 gate，loader 按 KV group 打包为 Megatron `linear_qkv`；在线转换执行逆变换。专家 tensor 通过 safetensors slice API 按 global expert id 读取，避免物化整组 expert。

在线同步在 TP/EP gather 前过滤 PLE table、三个 PLE 派生 buffer 和 QSA Indexer。Actor/ref 的 pinned-CPU snapshot 使用同一规则，避免在 TP=8 时每个训练 rank 再复制约 12 GiB PLE table。Tensor transport 与 distributed transport 使用同一过滤规则。SGLang 进程在启动时加载完整参数集合；optimizer step 后只接收 `trainable_sync`。每次同步携带单调递增 `weight_version`。P0 的在线传输限定为 tensor/NCCL；disk checkpoint 更新缺少冻结的 PLE 与 QSA Indexer tensor，创建 updater 时直接报错。

## SGLang Day-0 overlay

SGLang 固定提交应用两份独立补丁：

1. [`sglang-preserve-static-weight-reset.patch`](../../../docker/patch/qwen4-exp/sglang-preserve-static-weight-reset.patch)：给 PLE table 和 QSA Indexer 标记 `_preserve_on_weight_reset`，使 WeightChecker 的覆盖率探测保留启动值；同时把空 `seed` 解析为公开 checkpoint 的 1234。
2. [`sglang-qsa-short-extend.patch`](../../../docker/patch/qwen4-exp/sglang-qsa-short-extend.patch)：write plan 传递 `compress_plan_valid`；extend chunk 的固定容量尾项统一读取 source row 0。该补丁覆盖 1-token health probe：压缩比为 4 时，原尾项形成 `[0,1,2,3]`，输入仅有 1 行；有效位折叠后读取 `[0,0,0,0]`，真实完整 group 仍读取 `[0,1,2,3]`。

[`tools/qwen4_exp/build_environment.sh`](../../../tools/qwen4_exp/build_environment.sh) 从固定 PR commit 构建 SGLang，依次执行两份补丁的 `git apply --check`、Python 语法检查和 CPU write-plan 行为检查，再构建 slime image。容器 probe 校验 Transformers/Megatron/SGLang 版本以及两份 overlay 的运行时代码标记。

## 并行与 P0 启动参数

默认内网门禁使用 8 nodes × 8 GPUs：

| 侧 | 拓扑 |
|---|---|
| Megatron actor | world 64，TP=8，EP=8，PP=1，CP=1，ETP=1，sequence parallel |
| SGLang rollout | 1 engine × 64 GPUs，DP=64，EP=64，dense TP=1 |

训练 world 经 TP 划分后得到 8 个 data-parallel ranks；EP=8 使每个 expert group 的 512 个 experts 按 64 experts/rank 分布。启动脚本会检查 world/TP、DP/EP、expert/EP 的整除关系，并固定单 rollout engine，以便 checksum 对 64 个 engine ranks 做完整覆盖。

[`scripts/run-qwen4-exp-p0-validation.sh`](../../../scripts/run-qwen4-exp-p0-validation.sh) 运行两轮 GRPO：8 prompts、每 prompt 2 samples、global batch 16、最大 context 256、最大 response 32。交替 group reward 为每个 prompt group 产生确定性方差；学习率 `1e-4` 用于放大单步权重变化，验收目标是闭环可观察性。

## 一条命令验收

先构建固定环境：

```bash
tools/qwen4_exp/build_environment.sh
```

内网已有 SGLang 镜像仓库时，通过 `SGLANG_SOURCE=/path/to/sglang` 使用本地 source checkout；该 checkout 需要包含固定 commit。`QWEN4_EXP_BUILD_ROOT=/path/to/build` 保留应用 overlay 后的 SGLang 源码，方便在构建失败时直接检查中间状态。

本地 metadata 与 CPU/tiny-model 验收：

```bash
tools/qwen4_exp/validate.sh \
  --mode local \
  --config-json /path/to/Qwen3.8-Flash-Next/config.json \
  --index-json /path/to/Qwen3.8-Flash-Next/model.safetensors.index.json \
  --output-root /path/to/qwen4-exp-validation
```

内网 8×8 GPU 全链路验收：

```bash
RAY_ADDRESS=auto tools/qwen4_exp/validate.sh \
  --mode intranet \
  --hf-checkpoint /path/to/Qwen3.8-Flash-Next \
  --expected-nodes 8 \
  --expected-gpus 64 \
  --output-root /shared/qwen4-exp-validation
```

验收器按阶段写 JSONL、summary JSON、最终 `result.json`，并生成 `.tar.gz` 与 SHA-256：

| 错误码 | 阶段 | 关闭条件 |
|---|---|---|
| `Q4E-100` | preflight | CUDA、Ray 节点/GPU、共享目录、版本、源码 hash、SGLang API 与 overlay 一致 |
| `Q4E-200` | manifest | 1,658 tensors、131 files、总大小、四类数量与 manifest digest 精确匹配 |
| `Q4E-300` | unit | config、GR/GDN/PLE/QSA、packed、映射、同步过滤与验收器测试通过 |
| `Q4E-400` | E2E | 两轮 generate→reward→backward→step→sync→regenerate 命令成功 |
| `Q4E-410` | checkpoint alignment | 所有 64 个 Megatron ranks 均有 load audit；关键 HF→Megatron tensor 抽样逐值一致；PLE TP shard 完整加载 |
| `Q4E-500` | model transition | trainable 指纹变化；PLE/Indexer 与 derived buffer 指纹保持不变 |
| `Q4E-510` | engine sync | 64 个 SGLang ranks 全覆盖；trainable checksum 变化；static checksum 保持不变 |
| `Q4E-600` | RL closure | logprob 差异不超过阈值、gradient norm 有限且大于 0、reward 有方差、两轮权重版本不同 |
| `Q4E-900` | archive | 证据目录成功归档并写 checksum |

Transformers↔Megatron 的证据由三层组成：公开 manifest 全量分类、checkpoint load audit 的真实 tensor 对齐、基于固定 Transformers 参考语义的 tiny GR/PLE/QSA forward 与 gradient tests。Megatron↔SGLang 的证据由训练/rollout logprob 阈值、同步前后双方 fingerprint/checksum 与第二轮生成共同给出。

## P0 消融实验

对照组为提交 `8ad267e`，处理组只删减 P0 启动参数无法到达、没有生产调用者或形成重复配置来源的实现。checkpoint、prompt、manifest contract 和 SGLang 固定提交保持一致。

| 指标 | 对照组 | 处理组 |
|---|---:|---:|
| Qwen4-Exp 相关顶层 symbol | 101 | 90 |
| 公开顶层 symbol | 47 | 31 |
| `reference.py` 行数 | 658 | 487 |
| 非测试、非文档净代码变化 | 0 | -290 行 |
| 本地门禁 | 53 项通过 | 54 项通过 |
| lifecycle manifest digest | `00ae2f76...e56d` | `00ae2f76...e56d` |

删除项及证据：

- 长上下文 QSA Top-K oracle、selected-index 返回值和 chunk checkpoint：P0 的 `seq_length=256`，公开 `indexer_budget=2048`；处理组新增超预算拒绝测试。
- package re-export、生命周期派生属性、无调用 driver event、`index_block_topk` 测试型属性和单行包装函数：生产调用图中没有使用者。
- validation-dir 环境变量和 fingerprint-regex 环境变量：CLI 已提供唯一配置来源，fingerprint 集合属于验收 contract。
- packed `sequence_ids`：模型计算只读取 `cu_seqlens`、`positions`、`max_seqlen` 和样本起止位置。
- Transformers 旧注册分支：固定的本地版本与容器版本都支持 `exist_ok=True`。

保留项包括 Qwen4-Exp config adapter、参数生命周期 manifest、PLE 流式分片、训练/rollout fingerprint 以及两份 SGLang overlay。删除这些 module 后，配置归一化、静态参数过滤、大表加载或内网证据采集会扩散到多个调用点。

## 当前验证状态

本地已完成以下证据：

- 两份 SGLang patch 在固定提交上 clean apply；patch 后源码通过 Python 语法检查；1-token 与 valid+padding write-plan CPU 行为检查通过；
- 54 项 Qwen4-Exp 门禁测试覆盖 config/lifecycle、HF config fallback、GR/GDN/PLE/QSA、checkpoint mapping、完整 Megatron tiny packed graph、数据 padding/mask 与验收器；
- 完整 Megatron tiny graph 在单进程 Gloo 上完成 forward、loss、backward，并确认 PLE table 无梯度、QSA Indexer 无在线同步项；
- `git diff --check`、Python compile 与 shell syntax 纳入最终本地检查。

内网命令负责生成完整 checkpoint、多机 GPU、真实 SGLang rollout 和一步 RL 的运行时证据。验收结果以归档中的 `result.json`、`stages.jsonl` 及各阶段 summary 为准。
