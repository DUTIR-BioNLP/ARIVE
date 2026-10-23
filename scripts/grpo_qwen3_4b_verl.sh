#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

die() {
  echo "[grpo] ERROR: $*" >&2
  exit 1
}

has_hf_weights() {
  local checkpoint_dir="$1"
  [[ -f "${checkpoint_dir}/model.safetensors.index.json" \
    || -f "${checkpoint_dir}/model.safetensors" \
    || -f "${checkpoint_dir}/pytorch_model.bin.index.json" \
    || -f "${checkpoint_dir}/pytorch_model.bin" ]]
}

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] \
  || die "set CUDA_VISIBLE_DEVICES to the physical GPU ids to use"
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
gpu_count="${#visible_gpus[@]}"
(( gpu_count >= 4 )) \
  || die "the default rollout/reward layout requires at least four visible GPUs"

if [[ -n "${QWENVL_ENV_DIR:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${QWENVL_ENV_DIR}/bin/python}"
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
export MCOT_ABNORMAL_ANNOTATION_PATH="${MCOT_ABNORMAL_ANNOTATION_PATH:-${DATA_ROOT}/abnormal_mcot_annotation.json}"
export MCOT_ANNOTATION_PATH="${MCOT_ANNOTATION_PATH:-${DATA_ROOT}/mcot_annotation.json}"
export MIMIC_CXR_ROOT="${MIMIC_CXR_ROOT:-${DATA_ROOT}/images}"
[[ -d "${MIMIC_CXR_ROOT}" ]] || die "MIMIC_CXR_ROOT is not a directory: ${MIMIC_CXR_ROOT}"

DATASET_NAME="${MCOT_GRPO_DATASET_NAME:-ABN_MIMIC_COT}"
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

MODEL_PATH="${MCOT_BASE_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
REF_MODEL_PATH="${MCOT_REF_MODEL:-${MODEL_PATH}}"
SFT_MODEL_PATH="${MCOT_SFT_MODEL_PATH:-}"
VLLM_MODEL_PATH="${MCOT_VLLM_MODEL_PATH:-}"
OUTPUT_DIR="${MCOT_GRPO_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/grpo}"
[[ -d "${SFT_MODEL_PATH}" ]] || die "set MCOT_SFT_MODEL_PATH to the full SFT checkpoint"
has_hf_weights "${SFT_MODEL_PATH}" \
  || die "full SFT checkpoint has no supported Hugging Face weight files"
[[ -d "${VLLM_MODEL_PATH}" ]] \
  || die "set MCOT_VLLM_MODEL_PATH to the output of tools/export_vllm_checkpoint.py"
[[ -f "${VLLM_MODEL_PATH}/config.json" ]] || die "vLLM export has no config.json"
[[ -f "${VLLM_MODEL_PATH}/model.safetensors.index.json" ]] \
  || die "vLLM export has no model.safetensors.index.json"

NPROC_PER_NODE="${NPROC_PER_NODE:-${gpu_count}}"
NNODES="${NNODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 400))}"
[[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "NPROC_PER_NODE must be positive"
[[ "${MASTER_PORT}" =~ ^[1-9][0-9]*$ ]] && (( MASTER_PORT <= 65518 )) \
  || die "MASTER_PORT must be an integer in [1, 65518] (a second port uses +17)"
(( NPROC_PER_NODE <= gpu_count )) \
  || die "NPROC_PER_NODE=${NPROC_PER_NODE} exceeds visible GPUs=${gpu_count}"
[[ "${NNODES}" == "1" ]] || die "this launcher currently supports one node"

export QWENVL_GRPO_VLLM_MODEL_PATH="${VLLM_MODEL_PATH}"
export QWENVL_GRPO_VLLM_TP_SIZE="${QWENVL_GRPO_VLLM_TP_SIZE:-${gpu_count}}"
[[ "${QWENVL_GRPO_VLLM_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] \
  || die "QWENVL_GRPO_VLLM_TP_SIZE must be positive"
(( QWENVL_GRPO_VLLM_TP_SIZE <= gpu_count )) \
  || die "vLLM tensor parallel size exceeds visible GPUs"
(( NPROC_PER_NODE == 1 || QWENVL_GRPO_VLLM_TP_SIZE > 1 )) \
  || die "multi-rank shared rollout requires vLLM tensor parallel size > 1"
export QWENVL_GRPO_VLLM_OWNER_RANK="${QWENVL_GRPO_VLLM_OWNER_RANK:-0}"
export QWENVL_GRPO_VLLM_WEIGHT_TRANSFER_BACKEND="${QWENVL_GRPO_VLLM_WEIGHT_TRANSFER_BACKEND:-ipc}"
export QWENVL_GRPO_VLLM_SYNC_VERIFY="${QWENVL_GRPO_VLLM_SYNC_VERIFY:-0}"
export QWENVL_GRPO_SHARED_VISION_OWNER_ONLY="${QWENVL_GRPO_SHARED_VISION_OWNER_ONLY:-1}"
export QWENVL_GRPO_SHARED_PREPARE_ALL_RANKS="${QWENVL_GRPO_SHARED_PREPARE_ALL_RANKS:-1}"
export QWENVL_GRPO_VLLM_NCCL_MASTER_ADDR="${MASTER_ADDR}"
export QWENVL_GRPO_VLLM_NCCL_MASTER_PORT="$((MASTER_PORT + 17))"
export QWENVL_GRPO_VLLM_NCCL_PACKED="${QWENVL_GRPO_VLLM_NCCL_PACKED:-0}"
export QWENVL_GRPO_VLLM_GPU_MEM_UTIL="${QWENVL_GRPO_VLLM_GPU_MEM_UTIL:-0.06}"
export QWENVL_GRPO_VLLM_MM_PROCESSOR_CACHE_GB="${QWENVL_GRPO_VLLM_MM_PROCESSOR_CACHE_GB:-0}"
export QWENVL_GRPO_VLLM_LIMIT_MM_IMAGE="${QWENVL_GRPO_VLLM_LIMIT_MM_IMAGE:-256}"
export QWENVL_GRPO_VLLM_TOP_K="${QWENVL_GRPO_VLLM_TOP_K:--1}"
export QWENVL_GRPO_VLLM_REPETITION_PENALTY="${QWENVL_GRPO_VLLM_REPETITION_PENALTY:-1.0}"
export QWENVL_GRPO_VLLM_PRESENCE_PENALTY="${QWENVL_GRPO_VLLM_PRESENCE_PENALTY:-0.01}"
export QWENVL_GRPO_MAX_PAUSE_TIMES="${QWENVL_GRPO_MAX_PAUSE_TIMES:-18}"
export QWENVL_GRPO_MAX_CONSECUTIVE_DROPPED_BATCHES="${QWENVL_GRPO_MAX_CONSECUTIVE_DROPPED_BATCHES:-20}"
export QWENVL_GRPO_PAUSE_TOKEN_ID="${QWENVL_GRPO_PAUSE_TOKEN_ID:-151649}"
export QWENVL_GRPO_VLLM_SEED="${QWENVL_GRPO_VLLM_SEED:-3407}"
export QWENVL_ANSWER_CHEXBERT_DEVICE="${QWENVL_ANSWER_CHEXBERT_DEVICE:-cuda:1}"
export QWENVL_REASONING_CHEXBERT_DEVICE="${QWENVL_REASONING_CHEXBERT_DEVICE:-cuda:2}"
export QWENVL_RADGRAPH_DEVICE="${QWENVL_RADGRAPH_DEVICE:-cuda:3}"
export QWENVL_ENABLE_REASONING_CHEXBERT="${QWENVL_ENABLE_REASONING_CHEXBERT:-1}"
export QWENVL_ENABLE_RADGRAPH="${QWENVL_ENABLE_RADGRAPH:-1}"
export QWENVL_ALLOW_REWARD_FALLBACK="${QWENVL_ALLOW_REWARD_FALLBACK:-0}"
export QWENVL_ENABLE_PARALLEL_MODEL_REWARDS="${QWENVL_ENABLE_PARALLEL_MODEL_REWARDS:-1}"
export QWENVL_REWARD_PARALLEL_WORKERS="${QWENVL_REWARD_PARALLEL_WORKERS:-3}"
export QWENVL_REWARD_CACHE_MAX_SIZE="${QWENVL_REWARD_CACHE_MAX_SIZE:-20000}"
export QWENVL_GRPO_REWARD_WEIGHTS="${QWENVL_GRPO_REWARD_WEIGHTS:-0.25,0.25,0.25,0.25}"
export QWENVL_GRPO_VLLM_ALLOW_PARTIAL_LAYERWISE_RELOAD="${QWENVL_GRPO_VLLM_ALLOW_PARTIAL_LAYERWISE_RELOAD:-1}"
export QWENVL_GRPO_SYNC_INTERVAL_STEPS="${QWENVL_GRPO_SYNC_INTERVAL_STEPS:-1}"
# IPC transfers Python objects only between trusted local processes. Do not expose
# the rollout rendezvous endpoint to an untrusted network.
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"

if [[ "${QWENVL_ENABLE_REASONING_CHEXBERT}" == "1" ]]; then
  [[ -f "${QWENVL_REASONING_CHEXBERT_CKPT_PATH:-}" ]] \
    || die "set QWENVL_REASONING_CHEXBERT_CKPT_PATH to the reasoning classifier checkpoint"
fi
[[ -f "${QWENVL_CHEXBERT_CKPT_PATH:-}" ]] \
  || die "set QWENVL_CHEXBERT_CKPT_PATH to the CheXbert checkpoint"

resume="${MCOT_GRPO_RESUME_FROM_CHECKPOINT:-False}"
resume_requested=0
case "${resume,,}" in
  ""|false|0|no) ;;
  *) resume_requested=1 ;;
esac
if [[ -d "${OUTPUT_DIR}" && "${resume_requested}" == "0" ]]; then
  for candidate in "${OUTPUT_DIR}"/checkpoint-*; do
    [[ -d "${candidate}" ]] || continue
    checkpoint_name="${candidate##*/}"
    if [[ "${checkpoint_name}" =~ ^checkpoint-[0-9]+$ ]]; then
      die "output contains ${candidate}; request resume or choose a new output directory"
    fi
  done
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/preflight_grpo.py" \
  --policy "${MODEL_PATH}" \
  --reference "${REF_MODEL_PATH}" \
  --overlay "${SFT_MODEL_PATH}" \
  --vllm "${VLLM_MODEL_PATH}" \
  --visible-gpu-count "${gpu_count}" \
  --nproc "${NPROC_PER_NODE}" \
  --tp-size "${QWENVL_GRPO_VLLM_TP_SIZE}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --vllm-port "${QWENVL_GRPO_VLLM_NCCL_MASTER_PORT}" \
  || die "artifact/runtime preflight failed"

if [[ "${MCOT_GRPO_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[grpo] preflight passed"
  exit 0
fi

training_args=(
  --model_name_or_path "${MODEL_PATH}"
  --vllm_model_name_or_path "${VLLM_MODEL_PATH}"
  --ref_model_name_or_path "${REF_MODEL_PATH}"
  --dataset_use "${DATASET_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --run_name "${MCOT_GRPO_RUN_NAME:-qwen3vl-4b-arive-grpo}"
  --seed "${MCOT_GRPO_SEED:-3407}"
  --deepspeed "${PROJECT_ROOT}/configs/zero3.json"
  --bf16 True
  --learning_rate "${MCOT_GRPO_LR:-1e-6}"
  --weight_decay "${MCOT_GRPO_WEIGHT_DECAY:-0.0}"
  --per_device_train_batch_size "${MCOT_GRPO_BATCH_SIZE:-8}"
  --gradient_accumulation_steps "${MCOT_GRPO_GRAD_ACCUM:-5}"
  --dataloader_drop_last True
  --gradient_checkpointing True
  --num_train_epochs "${MCOT_GRPO_EPOCHS:-1}"
  --max_steps "${MCOT_GRPO_MAX_STEPS:--1}"
  --tune_mm_vision False
  --tune_mm_mlp False
  --tune_mm_llm True
  --lora_enable "${MCOT_GRPO_LORA_ENABLE:-False}"
  --lora_r "${MCOT_GRPO_LORA_R:-64}"
  --lora_alpha "${MCOT_GRPO_LORA_ALPHA:-128}"
  --lora_dropout "${MCOT_GRPO_LORA_DROPOUT:-0.0}"
  --use_local_vision_branch True
  --tune_mm_local_vision False
  --tune_mm_local_mlp False
  --enable_local_image True
  --local_branch_model_path "${SFT_MODEL_PATH}"
  --local_branch_strict_load True
  --num_generations "${MCOT_GRPO_NUM_GENERATIONS:-5}"
  --max_completion_length "${MCOT_GRPO_MAX_COMPLETION_LENGTH:-2048}"
  --temperature "${MCOT_GRPO_TEMPERATURE:-0.9}"
  --top_p "${MCOT_GRPO_TOP_P:-1.0}"
  --clip_eps "${MCOT_GRPO_CLIP_EPS:-0.2}"
  --beta_kl "${MCOT_GRPO_BETA_KL:-0.01}"
  --reward_eps "${MCOT_GRPO_REWARD_EPS:-1e-6}"
  --inner_epochs "${MCOT_GRPO_INNER_EPOCHS:-1}"
  --minibatch_size "${MCOT_GRPO_MINIBATCH_SIZE:-8}"
  --ref_logprob_micro_batch_size "${MCOT_GRPO_REF_LOGPROB_MICRO_BATCH_SIZE:-10}"
  --max_grad_norm "${MCOT_GRPO_MAX_GRAD_NORM:-1.0}"
  --log_completions "${MCOT_GRPO_LOG_COMPLETIONS:-False}"
  --rollout_backend_module qwenvl.rl_grpo_verl.user_rollout
  --rollout_backend_class UserRolloutBackend
  --reward_module qwenvl.rl_grpo_verl.rewards_parallel
  --reward_function_name compute_reward
  --logging_steps "${MCOT_GRPO_LOGGING_STEPS:-10}"
  --save_steps "${MCOT_GRPO_SAVE_STEPS:-200}"
  --save_total_limit "${MCOT_GRPO_SAVE_TOTAL_LIMIT:-15}"
  --save_every_steps "${MCOT_GRPO_SAVE_EVERY_STEPS:--1}"
  --max_prompt_samples "${MCOT_GRPO_MAX_PROMPT_SAMPLES:--1}"
  --rollout_num_workers "${MCOT_GRPO_ROLLOUT_WORKERS:-0}"
  --skip_final_save "${MCOT_GRPO_SKIP_FINAL_SAVE:-False}"
)

if [[ "${resume_requested}" == "1" ]]; then
  training_args+=(--resume_from_checkpoint "${resume}")
fi

echo "[grpo] policy=${MODEL_PATH}"
echo "[grpo] full SFT overlay=${SFT_MODEL_PATH} (strict local-vision validation enabled)"
echo "[grpo] vLLM checkpoint=${VLLM_MODEL_PATH}"
echo "[grpo] output=${OUTPUT_DIR}, processes=${NPROC_PER_NODE}"

exec "${TORCHRUN_BIN}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT_ROOT}/qwenvl/train/train_grpo_verl.py" \
  "${training_args[@]}"
