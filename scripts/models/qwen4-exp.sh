NLAYERS=48

MOE_LAYER_FREQ="["
for ((layer_idx=0; layer_idx<NLAYERS; layer_idx++)); do
  if (( layer_idx > 0 )); then
    MOE_LAYER_FREQ+=","
  fi
  MOE_LAYER_FREQ+="1"
done
MOE_LAYER_FREQ+="]"

MODEL_ARGS=(
  --spec "slime_plugins.models.qwen4_exp.model" "get_qwen4_exp_model_provider"

  --disable-bias-linear
  --group-query-attention
  --num-attention-heads 24
  --num-query-groups 2
  --kv-channels 256
  --num-layers 48
  --hidden-size 2560
  --ffn-hidden-size 640
  --seq-length 256

  --normalization RMSNorm
  --apply-layernorm-1p
  --position-embedding-type none
  --norm-epsilon 1e-6
  --rotary-percent 0.25
  --rotary-base 10000000
  --swiglu
  --untie-embeddings-and-output-weights
  --vocab-size 248320

  # Every text layer owns routed and shared experts.
  --moe-ffn-hidden-size 640
  --moe-shared-expert-intermediate-size 640
  --moe-router-score-function softmax
  --moe-token-dispatcher-type alltoall
  --moe-router-topk 10
  --moe-layer-freq "${MOE_LAYER_FREQ}"
  --num-experts 512
  --moe-grouped-gemm
  --moe-token-drop-policy probs
  --moe-permute-fusion
  --moe-aux-loss-coeff 0
  --moe-shared-expert-gate

  # Qwen4-Exp P0 graph contract.
  --qwen-gdn-backend "${QWEN4_EXP_GDN_BACKEND:-fla}"
  --freeze-indexer
)
