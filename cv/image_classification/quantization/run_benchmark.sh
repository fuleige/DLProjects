#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-formal}"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="${ROOT_DIR}/cv/image_classification/quantization/train.py"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/image_classification/torchao_quantization_runs}"
MODEL_NAME="${MODEL_NAME:-resnet18}"
BACKEND="${BACKEND:-x86_inductor}"
QAT_DEVICE="${QAT_DEVICE:-cuda}"
FLOAT_DEVICE="${FLOAT_DEVICE:-auto}"

COMMON_ARGS=(
  --mode compare
  --dataset-type cifar100
  --data-root "${DATA_ROOT}"
  --model-name "${MODEL_NAME}"
  --backend "${BACKEND}"
  --float-device "${FLOAT_DEVICE}"
  --qat-device "${QAT_DEVICE}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${MODE}" == "smoke" ]]; then
  PRESET_ARGS=(
    --float-epochs 1
    --qat-epochs 1
    --train-subset 512
    --val-subset 256
    --calib-batches 2
    --max-train-batches 2
    --max-val-batches 2
    --batch-size 32
    --num-workers 0
    --benchmark-warmup 2
    --benchmark-iters 3
  )
elif [[ "${MODE}" == "formal" ]]; then
  PRESET_ARGS=(
    --float-epochs 3
    --qat-epochs 1
    --train-subset 10000
    --val-subset 2000
    --calib-batches 20
    --batch-size 128
    --num-workers 4
    --benchmark-warmup 5
    --benchmark-iters 10
    --learning-rate 5e-4
  )
else
  echo "Unsupported mode: ${MODE}" >&2
  echo "Usage: $0 [smoke|formal] [extra train.py args...]" >&2
  exit 1
fi

echo "Running torchao quantization benchmark"
echo "  mode       : ${MODE}"
echo "  model      : ${MODEL_NAME}"
echo "  backend    : ${BACKEND}"
echo "  data_root  : ${DATA_ROOT}"
echo "  output_dir : ${OUTPUT_DIR}"
echo "  qat_device : ${QAT_DEVICE}"
if [[ "$#" -gt 0 ]]; then
  echo "  extra_args : $*"
fi
echo "  note       : final resolved output dir is also printed by train.py"

python3 "${SCRIPT_PATH}" \
  "${COMMON_ARGS[@]}" \
  "${PRESET_ARGS[@]}" \
  "$@"
