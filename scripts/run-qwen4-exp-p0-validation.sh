#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${HF_CHECKPOINT:?HF_CHECKPOINT must point to the complete Qwen4-Exp checkpoint}"
: "${QWEN4_EXP_VALIDATION_DIR:?QWEN4_EXP_VALIDATION_DIR must be a shared validation directory}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-8}"
TRAIN_EP_SIZE="${TRAIN_EP_SIZE:-8}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-64}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-64}"
ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-64}"
ROLLOUT_EP_SIZE="${ROLLOUT_EP_SIZE:-64}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.70}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
PROMPT_DATA="${PROMPT_DATA:-${REPO_ROOT}/tools/qwen4_exp/assets/prompts.jsonl}"
MAX_TRAIN_ROLLOUT_DIFF="${MAX_TRAIN_ROLLOUT_DIFF:-0.1}"

WORLD_SIZE=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
if (( WORLD_SIZE % TRAIN_TP_SIZE != 0 )); then
  echo "actor world size ${WORLD_SIZE} is not divisible by TP ${TRAIN_TP_SIZE}" >&2
  exit 20
fi
TRAIN_DP_SIZE=$((WORLD_SIZE / TRAIN_TP_SIZE))
if (( TRAIN_DP_SIZE % TRAIN_EP_SIZE != 0 )); then
  echo "training DP ${TRAIN_DP_SIZE} is not divisible by EP ${TRAIN_EP_SIZE}" >&2
  exit 21
fi
if (( 512 % TRAIN_EP_SIZE != 0 )); then
  echo "512 experts are not divisible by training EP ${TRAIN_EP_SIZE}" >&2
  exit 22
fi
if (( ROLLOUT_NUM_GPUS != ROLLOUT_NUM_GPUS_PER_ENGINE )); then
  echo "the P0 gate requires one rollout engine spanning all rollout GPUs" >&2
  exit 23
fi
if (( ROLLOUT_DP_SIZE != ROLLOUT_NUM_GPUS || ROLLOUT_EP_SIZE != ROLLOUT_NUM_GPUS )); then
  echo "the P0 rollout gate requires DP=EP=rollout GPU count" >&2
  exit 24
fi

test -f "${HF_CHECKPOINT}/config.json"
test -f "${HF_CHECKPOINT}/model.safetensors.index.json"
test -f "${PROMPT_DATA}"
mkdir -p "${QWEN4_EXP_VALIDATION_DIR}/rollout_data"

source "${SCRIPT_DIR}/models/qwen4-exp.sh"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --load "${HF_CHECKPOINT}"
  --ref-load "${HF_CHECKPOINT}"
  --no-load-optim
  --no-save-optim
)

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --loss-mask-type qwen4_exp
  --group-rm
  --custom-rm-path slime_plugins.models.qwen4_exp.validation_reward.alternating_group_reward
  --num-rollout 2
  --rollout-batch-size 8
  --n-samples-per-prompt 2
  --global-batch-size 16
  --rollout-max-prompt-len 192
  --rollout-max-context-len 256
  --rollout-max-response-len 32
  --rollout-temperature 0.7
  --rollout-top-p 1.0
  --rollout-stop-token-ids 248044
  --save-debug-rollout-data "${QWEN4_EXP_VALIDATION_DIR}/rollout_data/{rollout_id}.pt"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-rollout-logprobs
  --kl-loss-coef 0
  --kl-loss-type low_var_kl
  --kl-coef 0
  --entropy-coef 0
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-4
  --lr-warmup-iters 0
  --lr-decay-style constant
  --weight-decay 0
  --adam-beta1 0.9
  --adam-beta2 0.98
  --use-stateless-adam
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TRAIN_TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size "${TRAIN_EP_SIZE}"
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 512
  --data-pad-size-multiplier 1
  --log-probs-chunk-size 64
)

SGLANG_ARGS=(
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
  --sglang-server-concurrency 16
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-enable-dp-attention
  --sglang-enable-dp-lm-head
  --sglang-dp-size "${ROLLOUT_DP_SIZE}"
  --sglang-ep-size "${ROLLOUT_EP_SIZE}"
  --sglang-moe-dp-size 1
  --sglang-moe-dense-tp-size 1
  --sglang-moe-a2a-backend deepep
  --sglang-deepep-mode auto
  --sglang-page-size 64
  --sglang-kv-cache-dtype bfloat16
  --sglang-context-length 256
  --sglang-max-prefill-tokens 512
  --sglang-chunked-prefill-size 512
  --sglang-max-running-requests 16
  --sglang-language-model-only
  --sglang-ple-offload-embedding
  --sglang-linear-attn-backend triton
  --sglang-disable-radix-cache
  --sglang-disable-overlap-schedule
  --sglang-disable-prefill-cuda-graph
  --sglang-enable-deterministic-inference
  --sglang-watchdog-timeout 7200
  --sglang-dist-timeout 1800
)

MISC_ARGS=(
  --attention-dropout 0
  --hidden-dropout 0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --deterministic-mode
  --skip-eval-before-train
  --check-weight-update-equal
  --ci-test
  --ci-disable-kl-checker
  --ci-train-rollout-logprob-abs-diff-threshold "${MAX_TRAIN_ROLLOUT_DIFF}"
  --qwen4-exp-validation-dir "${QWEN4_EXP_VALIDATION_DIR}"
)

RUNTIME_ENV_JSON="$({
  SLIME_RUNTIME_ROOT="${REPO_ROOT}" \
  QWEN4_EXP_MEGATRON_ROOT="${MEGATRON_ROOT}" \
  python3 - <<'PY'
import json
import os

repo = os.environ["SLIME_RUNTIME_ROOT"]
megatron = os.environ["QWEN4_EXP_MEGATRON_ROOT"]
path = ":".join(item for item in (repo, megatron, os.environ.get("PYTHONPATH")) if item)
env = {
    "PYTHONPATH": path,
    "PYTHONUNBUFFERED": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NCCL_ALGO": "Tree",
    "NCCL_NVLS_ENABLE": "0",
    "NVSHMEM_DISABLE_NCCL": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
}
for name in ("MASTER_ADDR", "NO_PROXY", "no_proxy", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
    if os.environ.get(name):
        env[name] = os.environ[name]
print(json.dumps({"env_vars": env}, separators=(",", ":")))
PY
})"

ray job submit \
  --address="${RAY_DASHBOARD_ADDRESS}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u "${REPO_ROOT}/train.py" \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --colocate \
  --update-weight-mode full \
  --update-weight-transport nccl \
  --update-weight-buffer-size 2147483648 \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}"
