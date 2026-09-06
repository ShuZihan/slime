# Qwen4-Exp 在 slime 中的 RL 适配设计与交付

运行入口见 [Qwen4-Exp README](../../../tools/qwen4_exp/README.md)：直接拉取 `qwen4-exp-rl` 分支，复用已有 `slime:latest` 镜像。

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

## 后续功能路线

### v1.0 目标

Qwen4-Exp v1.0 以 Qwen3.5 已有配方级能力为基线，并补齐 Qwen4-Exp 特有的长序列 QSA。完成口径为：

- text 与 Vision GRPO 均能完成 `generate → reward → backward → step → online sync → regenerate`；
- 训练侧执行真实 QSA selection，先通过 4K，再支持 32K packed sequence；
- 训练拓扑支持 TP、EP、PP2、CP4 与 sequence parallel，v1.0 保持 `ETP=1`；
- SGLang rollout 使用 checkpoint 自带的 NEXTN/MTP draft，并由 EAGLE worker 完成 speculative decoding；
- Megatron 保持 BF16 训练，SGLang 支持 FP8 weight rollout 与独立的 FP8 KV cache；
- PLE embedding table 与 QSA Indexer 继续冻结，PLE projection、norm、gate、conv 以及 QSA 主 Q/K/V/O 路径参与训练和在线同步。

在线 MTP 训练、`ETP>1` 和 routed-expert INT4 属于 v1.0 之后的增强项。这个边界与 Qwen3.5 当前证据一致：Qwen3.5 已有 native MTP rollout、Vision、CP、PP 和通用 FP8 路径；在线 MTP、ETP 与模型专用 INT4 的运行证据较弱。

本文统一使用“native MTP/NEXTN + EAGLE worker”描述 Qwen4-Exp rollout：checkpoint 中的 `model.mtp.*` 提供单层 draft，SGLang 将 CLI 的 `NEXTN` 归一为 `EAGLE` worker。EAGLE3 作为独立 draft 方案单列，不纳入 Qwen4-Exp v1.0。

### 实施顺序

| 阶段 | 交付内容 | 核心机制 | 阶段验收 |
|---|---|---|---|
| M0 | 当前短序列 text P0 内网闭环 | 完整 checkpoint 加载；冻结 PLE table 与 Indexer；同步全部 `trainable_sync` 参数 | 64 GPU 两轮 rollout；gradient norm 有限且非零；trainable checksum 改变；static checksum 保持不变；生成 `result.json` 与证据归档 |
| M1 | rollout-only native MTP/NEXTN 与 PP2 | EAGLE worker 读取 checkpoint MTP，Megatron 暂不构建 MTP；PP 按全局 layer offset 构建本 stage 层，跨 stage 传递 `4H` mHC state；PLE 只在所属 stage 加载 | speculative on/off greedy token 与 target logprob 对齐；记录 acceptance rate/length 与 throughput；完成短请求、radix-cache churn 的 24 小时无重启 soak，acceptance 与吞吐无持续衰减；PP1/PP2 的 logits、loss、gradient 与在线权重名一致 |
| M2 | `CP=1` 的真实长序列 QSA | 复用 DSA 的 packed causal boundary、query chunking 和定宽 indices ABI；冻结 Indexer，执行 compress4、block Top-512、token 展开与 causal tail；主 Q/K/V/O 通过 sparse GQA forward/backward 训练 | 2048、2051、2052、4K、8K 与 32K packed 输入；selected indices、attention output、loss 和 gradient 对齐固定 Transformers 参考；运行时不构造完整 `S×S` attention mask |
| M3 | Vision RL | 复用 Qwen3.5 Vision tower；注入 image/video embeddings；为 QSA 生成 3D position IDs；同时保留原始 token IDs 供 PLE n-gram 使用；visual 参数进入 optimizer 与在线同步 | image/video 各一个 packed case；Vision 参数产生梯度并同步；Megatron 与 SGLang teacher-forced logprob 对齐；两轮 Vision GRPO 完成 |
| M4 | CP2 到 CP4 | 先以 DSA allgather-CP 建立正确性基线：QSA 保留 local query，收集全局 main K/V 与压缩 Indexer K，执行 exact Global Top-K；目标环境 profiling 确认容量或性能需求后，再增加 owner-sharded Indexer、Local Top-K、Global Top-K 与分布式 sparse GQA；GDN 复用 gather/compute/slice，PLE 重建 packed token/history 后切回本 rank | CP1/CP2/CP4 的 selected indices、token-level logits、loss、gradient 对齐；packed sample 间无状态泄漏；collective 次序在长序列和空 padding case 下稳定；分别记录 allgather 与 owner-sharded 路径的通信量、峰值显存和 step time |
| M5 | FP8 rollout | BF16 actor 在线转换并发送 FP8 trainable weights；首版保留 PLE table、Indexer、Vision 和 MTP 为 BF16；FP8 KV cache 作为独立开关 | BF16/FP8 rollout 的 greedy token、logprob 误差、QSA selected blocks 和在线更新前后 checksum 满足阈值 |
| M6 | 在线 MTP 训练 | MTP 接收最终 decoder 的 `4H` mHC hidden state；通过 `fc_embedding + fc_hidden` 融合 token embedding 与 hidden state；训练 MTP loss 并同步 draft 参数 | MTP loss 有限；MTP 参数在 step 后改变且到达所有 draft ranks；更新后 acceptance length 无异常退化 |
| M7 | 按需增加 ETP 与 INT4 | ETP 先完成 EP×ETP global expert ownership 和在线 gather；INT4 复用 Qwen3.5 fused 3D expert 到逐 expert 2D 的离线/在线布局，首版只量化 routed experts | ETP1/ETP2 expert 输出及梯度对齐；INT4 离线 checkpoint 与在线更新使用相同 global expert id、scale 和 pack ABI；完成多 GPU RL 闭环 |

M0 提供所有后续阶段的对照证据。M1 中 native MTP/NEXTN rollout 和 PP2 分成两个独立提交。M2 在 `CP=1` 下固定 QSA 的长序列语义；M3 在同一 position/indexer 接口上接入多模态；M4 先建立 allgather-CP 正确性基线，再根据目标环境数据决定是否进入 owner-sharded 性能子阶段。BF16 功能闭环完成后进入 M5，避免把模型语义偏差与量化误差混在同一次排查中。

### M2 的 QSA 语义

`indexer_budget=2048` 表示每个 query 最多从 Top-512 完整 blocks 展开 2048 个历史 tokens，模型总序列可以远大于 2048。当前 P0 将每条训练序列限制在 2048 以内，此时所有可见完整 blocks 都会入选，full-selection causal SDPA 与 QSA 主注意力等价。由于不足 4 tokens 的 causal tail 会直接附加，当 query 的可见上下文为 2051 tokens 时仍覆盖全部历史；2052 tokens 包含 513 个完整 blocks，首次触发 Top-512 裁剪。

超过预算后，每个 query 需要执行以下流程：

1. Indexer 从 hidden state 生成 4 个 query heads 与 1 个 key head，head dim 为 128；
2. 可见历史按连续 4 tokens 压缩为 block key，完整 block 内的 raw key 取均值，并使用 block 起始位置的 RoPE；
3. 计算 `relu(q_index · k_block).sum(heads) / sqrt(128)`，从完整 blocks 中选择 Top-512；
4. 把 block ids 展开为 token ids，并附加不足 4 tokens 的 causal tail；
5. 主 Q/K/V attention 只读取 selected tokens，随后应用 query gate 和 output projection。

Top-K indices 是离散选择结果。M2 保持 Indexer 冻结并在无梯度区域计算 selection；Q/K/V/O、mHC 和其余 trainable text 参数继续参与 autograd。Indexer 对所有 query 扫描各自可见的完整历史 blocks，完整序列的打分量为 `O(S² / 4)`；query chunking 将 score 峰值控制为 `O(query_chunk × S / 4)`，sparse GQA 的工作集为 `O(query_chunk × (2048 + 3))`，全程不分配完整 `S×S` mask。Packed samples 分别构造 causal history，Top-K、tail、RoPE 和 PLE 状态均不能跨越 `cu_seqlens` 边界。

### DSA 对 QSA 的直接复用边界

slime 的 DSA 训练已经具备长序列所需的骨架：每个 query 在完整 causal history 上执行真实 Top-2048，使用定宽 indices 驱动 SparseMLA forward/backward；packed `cu_seqlens` 隔离样本；query chunking 控制临时 score；allgather-CP 收集全局 Indexer K 和 MLA KV，使 local query 得到 exact Global Top-K。对应实现位于 [`glm5.py`](../../../slime_plugins/models/glm5/glm5.py)、[`ops/indexer.py`](../../../slime_plugins/models/glm5/ops/indexer.py) 和 [`ops/sparse_mla.py`](../../../slime_plugins/models/glm5/ops/sparse_mla.py)。

M2 从 DSA 复用以下结构：

- packed sample 的 causal `starts/ends` 与 `cu_seqlens` 边界；
- query chunking、定宽 indices 和无效位置 `-1` 的 ABI；
- selector 与可微主注意力分离的 autograd 结构；
- selected indices、attention output、gradient 和训推 logprob 的分层验收方式。

QSA 需要实现两项模型专有逻辑：

- selector：raw Indexer K 按 4 tokens 求均值、K norm、block 起点 RoPE、4-head `ReLU` score 求和、Top-512、block-to-token 展开和 causal tail；
- attention：支持 24Q/2KV、head dim 256、query gate 的 sparse GQA forward/backward。

DSA 的 SparseMLA kernel 使用 latent KV 与 absorbed projection 布局，无法直接承载 QSA 的标准 GQA 张量。M2 只增加 QSA selector 与 sparse GQA 两个专有模块，不建立通用稀疏注意力框架，也不增加 Indexer 训练目标。已验证的 DSA 路径同样冻结 Indexer；QSA 延续该参数生命周期，使训练侧与 SGLang rollout 使用同一组 checkpoint selector 参数。

M2 的边界验收固定如下：

| 可见上下文 | 验收目的 |
|---:|---|
| 2048 | 当前 P0 dense-equivalent 回归基线 |
| 2051 | 512 个完整 blocks 加 3-token tail，最后一个 full-selection case |
| 2052 | 513 个完整 blocks，首次发生 Top-512 裁剪 |
| 4096 / 8192 | 多次 block 竞争、packed 边界、forward/backward 与 logprob 对齐 |
| 32768 | v1.0 长序列目标与峰值显存验收 |

M4 首先复用 DSA allgather-CP：每个 rank 保留 local query，allgather 全局压缩 Indexer K 与 main K/V，再执行 QSA Global Top-512 和 sparse GQA。这条路径承担 CP1/CP2/CP4 正确性对照。allgather 会复制完整历史，Indexer 扫描与 KV 显存仍随全局序列增长。若目标环境 profiling 显示该路径限制 32K 容量或 step time，M4 再实现 `owner-sharded block K → Local Top-K(index, score) → exact Global Top-K → owner-local sparse GQA → LSE merge`；该性能子阶段与 M2 解耦。

### 复用与代码边界

直接复用 Qwen3.5 已有实现：

- GDN 的 packed `cu_seqlens` 和 CP gather/slice；
- Hugging Face Vision module 的加载、embedding 注入和权重直通同步；
- slime 通用 MTP label/loss plumbing；
- FP8/INT4 updater、compressed-tensors post-process 和 expert collective 基础设施。

Qwen4-Exp 专有实现继续放在 `slime_plugins/models/qwen4_exp/`：mHC、QSA、PLE、Qwen4 MTP input fusion 以及对应的参数生命周期。每个阶段只解除已经实现的 runtime guard，并同时提交模型图、checkpoint 映射、lifecycle、recipe 和验收器。当前阶段不增加通用 capability registry，也不为未进入路线的模型变体建立适配层。

### 统一完成口径

每个阶段需要同时具备四层证据：

1. 固定参考源码上的 tensor、数学语义、gradient 与 checkpoint round-trip 测试；
2. 完整 checkpoint 在目标 Megatron 拓扑上的加载和一次训练 step；
3. Megatron→SGLang 在线同步后的 checksum、teacher-forced logprob 和生成结果；
4. 对应目标拓扑的两轮 RL、collective 稳定性和证据归档。

代码与单元测试完成后标记为 implemented；配方完成完整 checkpoint smoke 后标记为 runnable；目标拓扑完成上述四层验收后标记为 validated。

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

最终 mixer 位于最后一次 MoE scatter 之后，看到的是各 rank 的局部 token。其三个参数显式标记 `sequence_parallel`，由 Megatron 的 gradient finalizer 在 TP group 内执行 SUM。层内 GR 在 gather 后处理完整序列，不加这个标记，避免重复归约。

### GDN

36 个 linear-attention layer 复用 slime 的 `Qwen3_5GatedDeltaNet` 与 varlen `cu_seqlens` 接口。Qwen4-Exp checkpoint 的 output gate 为 `sigmoid`；共享实现读取 `output_gate_type`，缺省模型继续读取 `hidden_act`。这项修改保持 Qwen3.5 配置行为，并使 Qwen4-Exp 与参考实现一致。

### QSA

QSA 将历史 key 每 4 tokens 压缩成一个 block key，公开配置的选择预算为 2048 tokens。P0 的训练序列固定为 256 tokens，因此选择集合覆盖完整 causal prefix，训练侧逐个 packed sample 调用 causal SDPA。Q/K/V/O 路径参与 autograd；Indexer 三组 checkpoint 参数保留在模型状态中并冻结，用于维持 Megatron 与 SGLang 的启动权重一致性。

训练实现只接受 `max_seqlen ≤ 2048`。模型 provider 同时约束 `seq_length`，运行时再次检查每个 packed sample。超出预算的输入立即返回明确错误。本分支不携带长上下文 Top-K oracle、selected-index 输出和 query-chunk recompute 路径。

### PLE

PLE 在第 2 个 decoder layer、attention 前注入。输入 token 生成 bigram/trigram hash ids，经约 51B 参数的 n-gram table 查表，再执行 key/value projection、query-key gate 和 dilation=3 的 depthwise short convolution。

PLE table 的 BF16 容量约 95 GiB。Megatron 使用 `VocabParallelEmbedding` 按行切到 TP ranks；HF loader 依次打开 128 个 source tensors，仅物化与本 rank 行区间相交的 source shard，并检查覆盖区间连续、完整。三个 hash metadata buffer 从 config 生成，随后与 checkpoint tensor 做逐值校验。

P0 冻结 table，projection、norm 与 convolution 继续训练。SGLang 启用 `ple_offload_embedding`，其启动权重与 Megatron 来自同一 checkpoint。

分布式保存时，decoder layer 显式调用嵌套 `VocabParallelEmbedding.sharded_state_dict`，使 PLE table 保留全局行数、TP row offset 和分片数；其余 PLE 参数与 buffer 仍按 replica 保存。Provider 同时固定 heterogeneous checkpoint keys，避免不同 GDN/QSA/PLE 层被误当成同构层堆叠。此前把 TP table 当 replica 写出的 checkpoint 可能已丢失其他 rank 的行片，不能靠新的分片描述恢复缺失数据。

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

### 第一轮：初始实现简化

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

### 第二轮：正确性修复后的结构消融（2026-09-06）

本轮对照组为 `07c3f9d` 加上未提交的最终 mixer 梯度归约、PLE checkpoint 和 HF 导出修复；不能直接用 `git show 07c3f9d` 代替该基线。修改前保存了逐文件源码、SHA-256 和工作区 diff。依次应用下列三组消融，每组先完成相关回归，再继续下一组。

| 组别 | 删除或简化 | 依据 | 验证结果 |
|---|---|---|---|
| A：未接入的兼容路径 | 删除融合 QKV pack/unpack、对应 TE attention 命名别名及独立根 mixer 别名；删除缺失切片接口时整表加载 expert 的 fallback | 当前 provider 使用独立 Q/K/V，最终 mixer 位于 `decoder.final_layernorm`；真实 `SafetensorReader` 提供切片接口 | 参数映射、真实 P0 QSA 参数往返、HF 文件导出及模型 shell 共 18 项通过 |
| B：重复 PLE 状态 | 不再额外持有初始化 layout 的 tensor 副本，也不再每次前向构建 runtime layout；哈希直接读取注册 buffer | 三个持久化 buffer 才是加载、设备迁移和 checkpoint 的有效状态；layout 构造器仍供初始化和 loader 计算全表形状 | reference、映射与导出共 22 项通过；每次 PLE 前向 layout 构造次数从 1 降为 0 |
| C：单次转发函数 | 删除 SP gather/scatter 包装及单用设备函数；RMSNorm 计算并入模块 forward；删除层内未消费的 `args`、`layer_type` 属性 | collective 直接写在全序列/分片切换位置；设备和 dtype 放置仍由多调用者共享的函数承担 | reference、模型层、模型 shell、真实 TP=2 finalizer 和 checkpoint 共 13 项通过 |

四个生产文件从合计 1,187 行减至 1,079 行，净减少 **108 行、7 个顶层辅助函数**，未新增生产类或配置开关。五个已移除的参数名增加明确拒绝测试；原来的融合 QKV helper 往返测试改为实例化当前 `Qwen4ExpQSA`，检查其实际参数集合的加载与导出。

与修改前源码快照进行 10 组 CPU 数值对照：FP32/BF16 下的层内与最终 mHC、PLE、短序列 QSA，以及通过 `load_state_dict(assign=True)` 替换 PLE hash buffer 的情况。使用相同参数、输入和上游梯度，检查 state-dict key/value、冻结标记、前向、输入梯度与所有参数梯度；前向和梯度的最大绝对差异均为 **0**。这是本轮实现前后的有限样例对照，不是与 Transformers/SGLang 的全模型数值对齐。

本地完整门禁从 **66 passed、1 skipped** 到 **71 passed、1 skipped**；新增的 5 项是上述拒绝测试。`git diff --check` 与改动文件 Python 语法检查通过。没有 GPU 环境，本轮不评价真实性能、真实 GDN/MoE 内核或 RL 训练效果。

保留的设计及原因：

- 最终 mixer 的 SP 梯度 SUM，以及 PLE 嵌套 embedding 的 sharded-state hook：已由分布式回归证明是正确性约束。
- `Qwen4ExpNGramLayout` 的初始化构造、P0 config/runtime 校验和生命周期 manifest：分别被模型/loader、provider、同步/导出/验收实际使用。
- HF exporter 的跨 bucket expert 聚合、rank 0 错误传播和最后的完整性检查：权重碎片可以跨 bucket 到达，多 rank 必须在同一阶段退出；删掉这些机制会破坏现有导出保证。
- 冻结 PLE/Indexer 的同步过滤和源 checkpoint 静态 tensor 补齐，以及 rollout 验收记录：它们承担实际内存限制、权重完整性和验证职责。

## 当前验证状态

本次源码复核后修复了最终 mixer 梯度归约、PLE checkpoint 分片和完整 HF 导出，并补充对应回归。长序列 QSA、Vision、PP/CP/ETP 扩展、MTP 与 FP8 未进入本次修改。

`--save-hf` 现在使用专用 Qwen4-Exp 导出路径：TP/EP 收集前排除冻结 PLE/Indexer 与派生 buffer；rank 0 将 live 逐专家 gate/up/down 权重按 global expert id 还原成原始 HF 的 `gate_up_proj[E,2I,H]` 和 `down_proj[E,H,I]`；随后从同一 `--hf-checkpoint` 逐 tensor 读取冻结表、Indexer、hash buffer 及未训练的 Vision/MTP，保持源配置所需的完整权重集合。保留这些源 tensor 不表示支持相应扩展能力的训练。

导出会检查 live tensor 的覆盖、重复、shape 和 expert parts，任何遗漏均在生成最终 index 前报错。源 checkpoint 必须仍可读取，且与训练启动时的冻结权重一致。该路径由单个 writer 写盘，CPU 暂存正在组装的 expert group；PLE 按原始 shard 逐个读取，不跨 TP 汇总约 95 GiB 的整表。在线 tensor/NCCL 同步仍使用原有逐专家格式与 static 过滤规则。

本地验证的证据边界：

正确性修复后的基线为 **66 passed、1 skipped**，第二轮结构消融后为 **71 passed、1 skipped**；跳过项是显式启用的四 GPU 测试。CPU 门禁包含 TP 分片与 HF 导出检查。

GitHub 交付前增加了 Megatron namespace package 的版本定位回归，本地检查合计 **72 项通过、1 项跳过**。其中 3 项 Gloo/多进程测试因沙箱限制共享内存和本机通信，改在沙箱外重跑并通过；跳过项仍是未启用的四 GPU 测试。当时提交的使用文档已通过 Bash/Python 语法和保存/恢复脚本参数检查，后续入口整理为 README。这些检查不包含真实 checkpoint、多机 GPU 或 RL 质量验收。

- 两份 SGLang patch 在固定提交上 clean apply；patch 后源码通过 Python 语法检查；1-token 与 valid+padding write-plan CPU 行为检查通过；
- 原有单进程 tiny packed graph 使用了替代 GDN 和 MLP，只证明模型连接、packed metadata 与部分梯度流，不能证明真实 GDN/MoE 或 RL loss 闭环；测试已据此更名；
- 新增 TP=2 CPU/Gloo 测试，通过真实 Megatron finalizer 将最终 mixer 的梯度与完整序列参考对照，同时检查层内 GR 不被错误标记；
- 新增 PLE checkpoint 测试，通过 GPT→block→layer 的真实 sharded-state 路径检查 TP 行片，再使用 MCore mapping/planner、PyTorch 同步 CPU writer 和 MCore loader 完成两进程保存/恢复；不覆盖生产 CUDA staging 与异步 IO worker；
- 新增 HF 导出文件往返测试，使用真实 safetensors、转换器与 loader；transport 输入由 fixture 提供，覆盖跨 chunk/out-of-order expert parts、冻结 PLE 跨 source shard 重载、完整 key 集合和损坏输入拒绝；
- `git diff --check`、Python compile 与 shell syntax 纳入最终本地检查。

[`tests/test_qwen4_exp_gpu.py`](../../../tests/test_qwen4_exp_gpu.py) 提供独立的四 GPU 入口：真实 FLA GDN、TE grouped MoE、TP=2/EP=2、MCore DDP/finalizer、合成 policy loss、SGD step 和更新后 forward。仅缩小模型并替换 config IO，没有替换算子。该入口尚未在 GPU 上执行，默认不纳入 CPU 门禁；设置环境变量后，缺失 GPU 或依赖会直接失败。

```bash
QWEN4_EXP_RUN_GPU_TESTS=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
  torchrun --standalone --nproc-per-node=4 \
  -m pytest tests/test_qwen4_exp_gpu.py -q -s
```

内网命令负责生成完整 checkpoint、多机 GPU、真实 SGLang rollout 和一步 RL 的运行时证据。验收结果以归档中的 `result.json`、`stages.jsonl` 及各阶段 summary 为准。
