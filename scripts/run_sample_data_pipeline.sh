#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAMPLE_ROOT="${MCOT_SAMPLE_ROOT:-${PROJECT_ROOT}/data/example_synthetic}"
INPUT_DIR="${SAMPLE_ROOT}/inputs"
GPT_OUTPUT_DIR="${SAMPLE_ROOT}/gpt_outputs"
WORK_DIR="${MCOT_SAMPLE_WORK_DIR:-${SAMPLE_ROOT}/work}"
OUTPUT_DIR="${MCOT_SAMPLE_OUTPUT_DIR:-${SAMPLE_ROOT}/output}"
EXPECTED="${SAMPLE_ROOT}/expected/mcot_annotation.json"
ANATOMY_OPTIONS=()
if [[ -f "${SAMPLE_ROOT}/.legacy_swapped_dimensions" ]]; then
  ANATOMY_OPTIONS+=(--legacy-swapped-dimensions)
fi

mkdir -p "${WORK_DIR}" "${OUTPUT_DIR}"

echo "[sample] replay Stage 1 output -> Stage 2 merge_reasoning"
"${PYTHON_BIN}" -m data_generation.merge_reasoning \
  --base "${INPUT_DIR}/base_records.json" \
  --reasoning "${GPT_OUTPUT_DIR}/reasoning_predictions.json" \
  --output "${WORK_DIR}/reasoning.json" \
  --overwrite

echo "[sample] Stage 3 prepare_anatomy_boxes"
"${PYTHON_BIN}" -m data_generation.prepare_anatomy_boxes \
  --input "${INPUT_DIR}/anatomy_boxes.json" \
  --image-root "${INPUT_DIR}/images" \
  --output "${WORK_DIR}/anatomy.json" \
  "${ANATOMY_OPTIONS[@]}" \
  --overwrite

echo "[sample] Stage 4 attach_anatomy_boxes"
"${PYTHON_BIN}" -m data_generation.attach_anatomy_boxes \
  --records "${WORK_DIR}/reasoning.json" \
  --anatomy "${WORK_DIR}/anatomy.json" \
  --output "${WORK_DIR}/reasoning_with_boxes.json" \
  --missing-output "${WORK_DIR}/missing_anatomy.json" \
  --overwrite

echo "[sample] replay Stage 5 output -> Stage 6 build_mcot_dataset"
"${PYTHON_BIN}" -m data_generation.build_mcot_dataset \
  --bbox-source "${WORK_DIR}/reasoning_with_boxes.json" \
  --predictions "${GPT_OUTPUT_DIR}/bbox_predictions.json" \
  --output "${WORK_DIR}/mcot.json" \
  --validate-coordinates \
  --overwrite

echo "[sample] Stage 7 add_object_refs"
"${PYTHON_BIN}" -m data_generation.add_object_refs \
  --mcot "${WORK_DIR}/mcot.json" \
  --bbx "${WORK_DIR}/reasoning_with_boxes.json" \
  --output "${WORK_DIR}/mcot_with_object_refs.json" \
  --overwrite

echo "[sample] Stage 8 convert_to_qwenvl"
"${PYTHON_BIN}" -m data_generation.convert_to_qwenvl \
  --mcot "${WORK_DIR}/mcot_with_object_refs.json" \
  --clinical-context "${INPUT_DIR}/clinical_context.json" \
  --reference-reports "${INPUT_DIR}/reference_reports.json" \
  --output "${OUTPUT_DIR}/mcot_annotation.json" \
  --overwrite

"${PYTHON_BIN}" - "${OUTPUT_DIR}/mcot_annotation.json" "${EXPECTED}" <<'PY'
import json
import sys
from pathlib import Path

actual_path = Path(sys.argv[1])
expected_path = Path(sys.argv[2])
actual = json.loads(actual_path.read_text(encoding="utf-8"))
if expected_path.is_file():
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit(f"sample output differs from {expected_path}")
print(f"[sample] complete: {len(actual)} annotation -> {actual_path}")
PY
