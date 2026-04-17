#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-formal}"
shift || true
EXTRA_ARGS=("$@")

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_PATH="${ROOT_DIR}/cv/image_classification/quantization/train.py"
PYTHON_BIN="${PYTHON:-python3}"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/image_classification/torchao_quantization_runs}"
MODEL_NAME="${MODEL_NAME:-resnet18}"
BACKEND="${BACKEND:-x86_inductor}"
QAT_DEVICE="${QAT_DEVICE:-cuda}"
FLOAT_DEVICE="${FLOAT_DEVICE:-cuda}"

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

EFFECTIVE_DATA_ROOT="${DATA_ROOT}"
EFFECTIVE_OUTPUT_DIR="${OUTPUT_DIR}"
EFFECTIVE_MODEL_NAME="${MODEL_NAME}"
EFFECTIVE_BACKEND="${BACKEND}"
EFFECTIVE_FLOAT_DEVICE="${FLOAT_DEVICE}"
EFFECTIVE_QAT_DEVICE="${QAT_DEVICE}"

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  index=0
  while [[ "${index}" -lt "${#EXTRA_ARGS[@]}" ]]; do
    arg="${EXTRA_ARGS[${index}]}"
    case "${arg}" in
      --data-root)
        if [[ $((index + 1)) -lt "${#EXTRA_ARGS[@]}" ]]; then
          EFFECTIVE_DATA_ROOT="${EXTRA_ARGS[$((index + 1))]}"
        fi
        index=$((index + 2))
        continue
        ;;
      --data-root=*)
        EFFECTIVE_DATA_ROOT="${arg#*=}"
        ;;
      --output-dir)
        if [[ $((index + 1)) -lt "${#EXTRA_ARGS[@]}" ]]; then
          EFFECTIVE_OUTPUT_DIR="${EXTRA_ARGS[$((index + 1))]}"
        fi
        index=$((index + 2))
        continue
        ;;
      --output-dir=*)
        EFFECTIVE_OUTPUT_DIR="${arg#*=}"
        ;;
      --model-name)
        if [[ $((index + 1)) -lt "${#EXTRA_ARGS[@]}" ]]; then
          EFFECTIVE_MODEL_NAME="${EXTRA_ARGS[$((index + 1))]}"
        fi
        index=$((index + 2))
        continue
        ;;
      --model-name=*)
        EFFECTIVE_MODEL_NAME="${arg#*=}"
        ;;
      --backend)
        if [[ $((index + 1)) -lt "${#EXTRA_ARGS[@]}" ]]; then
          EFFECTIVE_BACKEND="${EXTRA_ARGS[$((index + 1))]}"
        fi
        index=$((index + 2))
        continue
        ;;
      --backend=*)
        EFFECTIVE_BACKEND="${arg#*=}"
        ;;
      --float-device)
        if [[ $((index + 1)) -lt "${#EXTRA_ARGS[@]}" ]]; then
          EFFECTIVE_FLOAT_DEVICE="${EXTRA_ARGS[$((index + 1))]}"
        fi
        index=$((index + 2))
        continue
        ;;
      --float-device=*)
        EFFECTIVE_FLOAT_DEVICE="${arg#*=}"
        ;;
      --qat-device)
        if [[ $((index + 1)) -lt "${#EXTRA_ARGS[@]}" ]]; then
          EFFECTIVE_QAT_DEVICE="${EXTRA_ARGS[$((index + 1))]}"
        fi
        index=$((index + 2))
        continue
        ;;
      --qat-device=*)
        EFFECTIVE_QAT_DEVICE="${arg#*=}"
        ;;
    esac
    index=$((index + 1))
  done
fi

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
echo "  model      : ${EFFECTIVE_MODEL_NAME}"
echo "  backend    : ${EFFECTIVE_BACKEND}"
echo "  data_root  : ${EFFECTIVE_DATA_ROOT}"
echo "  output_dir : ${EFFECTIVE_OUTPUT_DIR}"
echo "  float_dev  : ${EFFECTIVE_FLOAT_DEVICE}"
echo "  qat_device : ${EFFECTIVE_QAT_DEVICE}"
echo "  python     : ${PYTHON_BIN}"
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  echo "  extra_args : ${EXTRA_ARGS[*]}"
fi

"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  "${COMMON_ARGS[@]}" \
  "${PRESET_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
