"""Convert grounded ARIVE records to Qwen-VL conversation annotations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from project_paths import (
    CLINICAL_CONTEXT,
    MCOT_ANNOTATION,
    MCOT_WITH_OBJECT_REFS,
    REFERENCE_REPORTS,
)

from .box_utils import BOX_RE, OBJECT_REF_BOX_RE, coordinates, format_box
from .common import (
    context_value,
    dump_json,
    image_list,
    load_json_object,
    load_records,
    record_id,
    report_text,
    study_id_from_path,
)
from .generate_reasoning import build_report_index


SYSTEM_PROMPT = (
    "You are an assistant in radiology, responsible for analyzing medical imaging "
    "studies and generating detailed, structured, and accurate radiology reports."
)
MCOT_INSTRUCTION = (
    "Before answering, think step by step internally. In your internal thinking, "
    "output the bounding box of the key diagnostic region in the format "
    "<|object_ref_start|>anatomical object name<|object_ref_end|>"
    "<|box_start|>x1 y1 x2 y2<|box_end|>. Generate a comprehensive and detailed "
    "description of findings based on the current chest X-ray image and the indication."
)
MCOT_PRIOR_INSTRUCTION = (
    f"{MCOT_INSTRUCTION} Include a thorough comparison with a prior chest X-ray, "
    "emphasizing any significant changes, progression, or improvement."
)


def _reasoning_with_crop_observations(reasoning: str) -> str:
    def replace(match):
        coords = tuple(int(match.group(index)) for index in range(1, 5))
        return f"{format_box(coords)}\n<image>\n"

    return BOX_RE.sub(replace, reasoning)


def convert_record(
    record: dict[str, Any],
    clinical_context: dict[str, Any],
    prior_reports: dict[str, str],
    *,
    require_object_refs: bool,
) -> dict[str, Any]:
    images = image_list(record)
    if not images:
        raise ValueError("ARIVE record has no image")
    current_image = images[0]
    prior_image = images[1] if len(images) > 1 else current_image
    reasoning = record.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise KeyError("ARIVE record has no non-empty reasoning")

    boxes = coordinates(reasoning)
    if require_object_refs and len(OBJECT_REF_BOX_RE.findall(reasoning)) != len(boxes):
        raise ValueError(f"id={record_id(record)} has a box without an adjacent object reference")
    crop_images = [
        f"{current_image}_{x1}_{y1}_{x2}_{y2}" for x1, y1, x2, y2 in boxes
    ]
    assistant_reasoning = _reasoning_with_crop_observations(reasoning)

    current_id = study_id_from_path(current_image)
    raw_context = clinical_context.get(current_id)
    if not isinstance(raw_context, dict):
        raise KeyError(f"clinical context is missing study {current_id}")
    # Other clinical fields inform reasoning generation only. Match the
    # indication-only clinical input used at inference.
    indication = context_value(raw_context, "Indication")
    context_block = f"Indication: {indication}\n" if indication else ""

    if prior_image != current_image:
        prior_id = study_id_from_path(prior_image)
        if prior_id not in prior_reports:
            raise KeyError(f"prior report is missing study {prior_id}")
        user_text = (
            f"{SYSTEM_PROMPT}\n\n{context_block}Prior chest X-ray:\n<image>\n"
            f"Prior findings:{prior_reports[prior_id]}\nCurrent chest X-ray:\n<image>\n"
            f"{MCOT_PRIOR_INSTRUCTION}"
        )
        annotation_images = [prior_image, current_image, *crop_images]
    else:
        user_text = (
            f"{SYSTEM_PROMPT}\n\n{context_block}Current chest X-ray:\n<image>\n"
            f"{MCOT_INSTRUCTION}"
        )
        annotation_images = [current_image, *crop_images]

    assistant_text = f"<think>\n{assistant_reasoning}\n</think>\n{report_text(record)}"
    expected_images = user_text.count("<image>") + assistant_text.count("<image>")
    if expected_images != len(annotation_images):
        raise RuntimeError(
            f"id={record_id(record)} image/tag mismatch: "
            f"paths={len(annotation_images)}, tags={expected_images}"
        )
    return {
        "id": record_id(record),
        "image": annotation_images,
        "conversations": [
            {"from": "human", "value": user_text},
            {"from": "gpt", "value": assistant_text},
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcot", type=Path, default=MCOT_WITH_OBJECT_REFS)
    parser.add_argument("--clinical-context", type=Path, default=CLINICAL_CONTEXT)
    parser.add_argument("--reference-reports", type=Path, default=REFERENCE_REPORTS)
    parser.add_argument("--output", type=Path, default=MCOT_ANNOTATION)
    parser.add_argument("--allow-missing-object-refs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = load_json_object(args.clinical_context)
    reports = build_report_index(load_records(args.reference_reports))
    converted = [
        convert_record(
            record,
            context,
            reports,
            require_object_refs=not args.allow_missing_object_refs,
        )
        for record in load_records(args.mcot)
    ]
    dump_json(args.output, converted, overwrite=args.overwrite)
    print(f"wrote {len(converted)} annotations to {args.output}")


if __name__ == "__main__":
    main()
