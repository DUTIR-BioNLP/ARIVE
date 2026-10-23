#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

die() {
  echo "[sft] ERROR: $*" >&2
  exit 1
}

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] \
  || die "set CUDA_VISIBLE_DEVICES to the physical GPU ids to use"
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
gpu_count="${#visible_gpus[@]}"
(( gpu_count > 0 )) || die "CUDA_VISIBLE_DEVICES is empty"

if [[ -n "${QWENVL_ENV_DIR:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${QWENVL_ENV_DIR}/bin/python}"
  [[ -x "${QWENVL_ENV_DIR}/bin/torchrun" ]] \
    || die "QWENVL_ENV_DIR has no executable bin/torchrun"
  TORCHRUN_BIN="${TORCHRUN_BIN:-${QWENVL_ENV_DIR}/bin/torchrun}"
  export PATH="${QWENVL_ENV_DIR}/bin:${PATH}"
  if [[ -d "${QWENVL_ENV_DIR}/lib" ]]; then
    export LD_LIBRARY_PATH="${QWENVL_ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
  TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun || true)}"
fi
[[ -x "${PYTHON_BIN}" ]] || die "python is not available"
[[ -x "${TORCHRUN_BIN}" ]] || die "torchrun is not available"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
DATA_ROOT="${MCOT_DATA_ROOT:-${PROJECT_ROOT}/data}"
export MCOT_ANNOTATION_PATH="${MCOT_ANNOTATION_PATH:-${DATA_ROOT}/mcot_annotation.json}"
export MCOT_ABNORMAL_ANNOTATION_PATH="${MCOT_ABNORMAL_ANNOTATION_PATH:-${DATA_ROOT}/abnormal_mcot_annotation.json}"
export MIMIC_CXR_ROOT="${MIMIC_CXR_ROOT:-${DATA_ROOT}/images}"
[[ -d "${MIMIC_CXR_ROOT}" ]] || die "MIMIC_CXR_ROOT is not a directory: ${MIMIC_CXR_ROOT}"

MODEL_PATH="${MCOT_BASE_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
OUTPUT_DIR="${MCOT_SFT_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sft}"
DATASET_NAME="${MCOT_SFT_DATASET_NAME:-MIMIC_COT}"
IFS=',' read -r -a dataset_specs <<< "${DATASET_NAME}"
for dataset_spec in "${dataset_specs[@]}"; do
  dataset_key="${dataset_spec%%\%*}"
  case "${dataset_key}" in
    MIMIC_COT|MCOT)
      [[ -f "${MCOT_ANNOTATION_PATH}" ]] \
        || die "annotation file not found: ${MCOT_ANNOTATION_PATH}"
      ;;
    ABN_MIMIC_COT|MCOT_ABNORMAL)
      [[ -f "${MCOT_ABNORMAL_ANNOTATION_PATH}" ]] \
        || die "abnormal annotation file not found: ${MCOT_ABNORMAL_ANNOTATION_PATH}"
      ;;
    *) die "unknown packaged dataset name: ${dataset_key}" ;;
  esac
done
NPROC_PER_NODE="${NPROC_PER_NODE:-${gpu_count}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 500))}"
NNODES="${NNODES:-1}"

[[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "NPROC_PER_NODE must be positive"
[[ "${MASTER_PORT}" =~ ^[1-9][0-9]*$ ]] && (( MASTER_PORT <= 65535 )) \
  || die "MASTER_PORT must be an integer in [1, 65535]"
(( NPROC_PER_NODE <= gpu_count )) \
  || die "NPROC_PER_NODE=${NPROC_PER_NODE} exceeds visible GPUs=${gpu_count}"

"${PYTHON_BIN}" -c \
  "import deepspeed, flash_attn; from transformers import Qwen3VLForConditionalGeneration" \
  || die "required training packages are not importable"

if [[ "${MCOT_SFT_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[sft] preflight passed"
  exit 0
fi

training_args=(
  --deepspeed "${PROJECT_ROOT}/configs/zero3.json"
  --model_name_or_path "${MODEL_PATH}"
  --dataset_use "${DATASET_NAME}"
  --data_flatten True
  --tune_mm_vision True
  --tune_mm_mlp True
  --tune_mm_llm True
  --use_local_vision_branch True
  --tune_mm_local_vision True
  --tune_mm_local_mlp True
  --enable_local_image True
  --bf16 True
  --output_dir "${OUTPUT_DIR}"
  --num_train_epochs "${MCOT_SFT_EPOCHS:-6.0}"
  --per_device_train_batch_size "${MCOT_SFT_BATCH_SIZE:-8}"
  --per_device_eval_batch_size "${MCOT_SFT_EVAL_BATCH_SIZE:-16}"
  --gradient_accumulation_steps "${MCOT_SFT_GRAD_ACCUM:-1}"
  --max_pixels "${MCOT_MAX_PIXELS:-50176}"
  --min_pixels "${MCOT_MIN_PIXELS:-784}"
  --eval_strategy no
  --save_strategy epoch
  --save_total_limit "${MCOT_SFT_SAVE_TOTAL_LIMIT:-3}"
  --learning_rate "${MCOT_SFT_LR:-1e-5}"
  --weight_decay "${MCOT_SFT_WEIGHT_DECAY:-0}"
  --warmup_ratio "${MCOT_SFT_WARMUP_RATIO:-0.03}"
  --max_grad_norm "${MCOT_SFT_MAX_GRAD_NORM:-1}"
  --lr_scheduler_type cosine
  --logging_steps "${MCOT_SFT_LOGGING_STEPS:-1}"
  --model_max_length "${MCOT_SFT_MODEL_MAX_LENGTH:-4096}"
  --gradient_checkpointing True
  --dataloader_num_workers "${MCOT_DATALOADER_WORKERS:-8}"
  --run_name "${MCOT_SFT_RUN_NAME:-qwen3vl-4b-arive-sft}"
  --report_to "${MCOT_REPORT_TO:-none}"
)

echo "[sft] model=${MODEL_PATH}"
echo "[sft] dataset=${DATASET_NAME}"
echo "[sft] output=${OUTPUT_DIR}, processes=${NPROC_PER_NODE}"

exec "${TORCHRUN_BIN}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT_ROOT}/qwenvl/train/train_qwen.py" \
  "${training_args[@]}"
