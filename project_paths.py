"""Machine-neutral default paths for ARIVE data files.

All runtime data lives below ``data/`` by default.  Set ``MCOT_DATA_ROOT`` to
override the entire tree without changing individual commands.  A relative
override is resolved from the repository root, not from the caller's working
directory.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _data_root() -> Path:
    configured = os.environ.get("MCOT_DATA_ROOT")
    if not configured:
        return PROJECT_ROOT / "data"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _path_from_env(name: str, default: Path) -> Path:
    configured = os.environ.get(name)
    if not configured:
        return default
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


DATA_ROOT = _data_root()
SOURCE_DATA_DIR = DATA_ROOT / "source"
INFERENCE_DATA_DIR = DATA_ROOT / "inference"
IMAGE_ROOT = _path_from_env("MIMIC_CXR_ROOT", DATA_ROOT / "images")
PIPELINE_WORK_DIR = _path_from_env(
    "MCOT_PIPELINE_WORK_DIR", DATA_ROOT / "work" / "pipeline"
)

BASE_RECORDS = _path_from_env(
    "MCOT_BASE_RECORDS", SOURCE_DATA_DIR / "base_records.json"
)
FEW_SHOT_EXAMPLES = _path_from_env(
    "FEW_SHOT_EXAMPLES", SOURCE_DATA_DIR / "few_shot_examples.json"
)
REFERENCE_REPORTS = _path_from_env(
    "MCOT_REFERENCE_REPORTS", SOURCE_DATA_DIR / "reference_reports.json"
)
CLINICAL_CONTEXT = _path_from_env(
    "MCOT_CLINICAL_CONTEXT", SOURCE_DATA_DIR / "clinical_context.json"
)
ANATOMY_SOURCE = _path_from_env(
    "MCOT_ANATOMY_SOURCE", SOURCE_DATA_DIR / "anatomy_boxes.json"
)

REASONING_SHARD = PIPELINE_WORK_DIR / "reasoning" / "generated.json"
REASONING = PIPELINE_WORK_DIR / "reasoning.json"
NORMALIZED_ANATOMY = PIPELINE_WORK_DIR / "anatomy.json"
REASONING_WITH_BOXES = PIPELINE_WORK_DIR / "reasoning_with_boxes.json"
MISSING_ANATOMY = PIPELINE_WORK_DIR / "missing_anatomy.json"
BBOX_SHARD = PIPELINE_WORK_DIR / "bbox" / "generated.json"
MCOT = PIPELINE_WORK_DIR / "mcot.json"
MCOT_WITH_OBJECT_REFS = PIPELINE_WORK_DIR / "mcot_with_object_refs.json"

MCOT_ANNOTATION = _path_from_env(
    "MCOT_ANNOTATION_PATH", DATA_ROOT / "mcot_annotation.json"
)
ABNORMAL_MCOT_ANNOTATION = _path_from_env(
    "MCOT_ABNORMAL_ANNOTATION_PATH", DATA_ROOT / "abnormal_mcot_annotation.json"
)

INFERENCE_REPORTS = INFERENCE_DATA_DIR / "evaluation_reports.jsonl"
INFERENCE_REFERENCE_REPORTS = INFERENCE_DATA_DIR / "reference_reports.jsonl"
INFERENCE_CLINICAL_CONTEXT = INFERENCE_DATA_DIR / "clinical_context.json"
INFERENCE_ABN_REPORTS = INFERENCE_DATA_DIR / "mimic_abn_evaluation_reports.jsonl"
INFERENCE_ABN_ANNOTATION = (
    INFERENCE_DATA_DIR / "priorrg_mimic_abn_annotation.json"
)
INFERENCE_PRIOR_ANNOTATION = (
    INFERENCE_DATA_DIR / "priorrg_mimic_cxr_annotation.json"
)
