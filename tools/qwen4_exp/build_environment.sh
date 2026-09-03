#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SGLANG_COMMIT="78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385"
SGLANG_SHORT="${SGLANG_COMMIT:0:7}"
TRANSFORMERS_VERSION="5.12.1"
CUDA_VERSION="${CUDA_VERSION:-13.0.3}"
CUDA_TAG="cu$(printf '%s' "${CUDA_VERSION}" | tr -d '.' | cut -c1-3)"
SGLANG_IMAGE_TAG="${SGLANG_IMAGE_TAG:-qwen4-exp-${SGLANG_SHORT}-${CUDA_TAG}}"
SLIME_IMAGE="${SLIME_IMAGE:-slimerl/slime:qwen4-exp-${SGLANG_SHORT}-${CUDA_TAG}}"
PATCH_PATHS=(
  "${REPO_ROOT}/docker/patch/qwen4-exp/sglang-preserve-static-weight-reset.patch"
  "${REPO_ROOT}/docker/patch/qwen4-exp/sglang-qsa-short-extend.patch"
)

command -v git >/dev/null
command -v docker >/dev/null
for patch_path in "${PATCH_PATHS[@]}"; do
  test -f "${patch_path}"
done

if [[ -n "${QWEN4_EXP_BUILD_ROOT:-}" ]]; then
  BUILD_ROOT="${QWEN4_EXP_BUILD_ROOT}"
  mkdir -p "${BUILD_ROOT}"
  CLEAN_BUILD_ROOT=0
else
  BUILD_ROOT="$(mktemp -d -t qwen4-exp-build.XXXXXXXX)"
  CLEAN_BUILD_ROOT=1
fi

cleanup() {
  if [[ "${CLEAN_BUILD_ROOT}" == "1" ]]; then
    rm -rf -- "${BUILD_ROOT}"
  fi
}
trap cleanup EXIT

SGLANG_BUILD_DIR="${BUILD_ROOT}/sglang"
if [[ -e "${SGLANG_BUILD_DIR}" ]]; then
  echo "Build directory already contains ${SGLANG_BUILD_DIR}" >&2
  exit 2
fi

if [[ -n "${SGLANG_SOURCE:-}" ]]; then
  test -d "${SGLANG_SOURCE}/.git"
  git clone --no-hardlinks "${SGLANG_SOURCE}" "${SGLANG_BUILD_DIR}"
else
  git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang.git "${SGLANG_BUILD_DIR}"
  git -C "${SGLANG_BUILD_DIR}" fetch --no-tags origin refs/pull/36497/head
fi
git -C "${SGLANG_BUILD_DIR}" checkout --detach "${SGLANG_COMMIT}"

ACTUAL_SGLANG_COMMIT="$(git -C "${SGLANG_BUILD_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_SGLANG_COMMIT}" != "${SGLANG_COMMIT}" ]]; then
  echo "SGLang commit mismatch: ${ACTUAL_SGLANG_COMMIT}" >&2
  exit 3
fi
if [[ -n "$(git -C "${SGLANG_BUILD_DIR}" status --porcelain)" ]]; then
  echo "Copied SGLang checkout is dirty before applying the Qwen4-Exp patch" >&2
  exit 4
fi

for patch_path in "${PATCH_PATHS[@]}"; do
  git -C "${SGLANG_BUILD_DIR}" apply --check "${patch_path}"
  git -C "${SGLANG_BUILD_DIR}" apply "${patch_path}"
done
python3 -m py_compile \
  "${SGLANG_BUILD_DIR}/python/sglang/srt/layers/attention/qsa/metadata.py" \
  "${SGLANG_BUILD_DIR}/python/sglang/srt/layers/attention/qsa/qsa_indexer.py" \
  "${SGLANG_BUILD_DIR}/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py" \
  "${SGLANG_BUILD_DIR}/python/sglang/srt/models/qwen4_exp.py" \
  "${SGLANG_BUILD_DIR}/python/sglang/srt/utils/weight_checker.py"

docker build \
  --file "${SGLANG_BUILD_DIR}/docker/Dockerfile" \
  --build-arg BRANCH_TYPE=local \
  --build-arg BUILD_TYPE=all \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg SGLANG_BUILD_COMMIT="${SGLANG_COMMIT}" \
  --build-arg SGLANG_BUILD_URL="https://github.com/sgl-project/sglang/pull/36497" \
  --build-arg SGLANG_IMAGE_TAG="slimerl/sglang:${SGLANG_IMAGE_TAG}" \
  --tag "slimerl/sglang:${SGLANG_IMAGE_TAG}" \
  "${SGLANG_BUILD_DIR}"

docker run --rm \
  --volume "${SCRIPT_DIR}/check_sglang_patch.py:/tmp/check_sglang_patch.py:ro" \
  "slimerl/sglang:${SGLANG_IMAGE_TAG}" \
  python /tmp/check_sglang_patch.py --sglang-root /sgl-workspace/sglang

docker build \
  --file "${REPO_ROOT}/docker/Dockerfile" \
  --build-arg SGLANG_IMAGE_TAG="${SGLANG_IMAGE_TAG}" \
  --build-arg ENABLE_SGLANG_PATCH=0 \
  --build-arg SLIME_SOURCE=local \
  --build-arg TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION}" \
  --tag "${SLIME_IMAGE}" \
  "${REPO_ROOT}"

docker run --rm "${SLIME_IMAGE}" python -c \
  'import os, subprocess; from pathlib import Path; import megatron, sglang, slime, transformers; assert transformers.__version__ == "5.12.1"; assert os.environ["SGLANG_BUILD_COMMIT"] == "78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385"; assert subprocess.check_output(["git", "-C", "/root/Megatron-LM", "rev-parse", "HEAD"], text=True).strip() == "1dcf0dafa884ad52ffb243625717a3471643e087"; root=Path("/sgl-workspace/sglang/python/sglang/srt"); source=(root / "models/qwen4_exp.py").read_text(); checker=(root / "utils/weight_checker.py").read_text(); metadata=(root / "layers/attention/qsa/metadata.py").read_text(); indexer=(root / "layers/attention/qsa/qsa_indexer.py").read_text(); backend=(root / "layers/attention/qwen_sparse_attn_backend.py").read_text(); assert "_preserve_on_weight_reset = True" in source; assert "getattr(self.config, \"seed\", None) or 1234" in source; assert "_preserve_on_weight_reset" in checker; assert "compress_plan_valid" in metadata and "compress_plan_valid" in indexer and "group_plan_valid" in backend; print("Qwen4-Exp environment import, pin, and patch probe passed")'

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${SLIME_IMAGE}")"
printf 'Qwen4-Exp image: %s\n' "${SLIME_IMAGE}"
printf 'Image ID: %s\n' "${IMAGE_ID}"
printf 'SGLang commit: %s\n' "${SGLANG_COMMIT}"
printf 'Transformers version: %s\n' "${TRANSFORMERS_VERSION}"
