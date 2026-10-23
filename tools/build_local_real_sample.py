#!/usr/bin/env python3
"""Extract one linked, real local record into the private ``data/sample`` tree.

This utility is deliberately source-agnostic: every restricted source path is
supplied on the command line, and the resulting sample directory stays ignored
by git.  It extracts only the selected record, its prior report and clinical
context, the matching raw anatomy row, real few-shot records, the two model
outputs needed by the deterministic replay script, and only the selected
current/prior images.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sample"


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            value = [json.loads(line) for line in handle if line.strip()]
    else:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TypeError(f"{path} must contain a JSON list or JSONL objects")
    return value


def load_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def record_id(record: dict[str, Any]) -> Any:
    for key in ("id", "question_id"):
        if key in record:
            return record[key]
    raise KeyError("record has neither id nor question_id")


def images(record: dict[str, Any]) -> list[str]:
    value = record.get("image")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise TypeError("record image must be a string or list of strings")


def report_text(record: dict[str, Any]) -> str:
    for key in ("report", "findings"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for turn in reversed(conversations):
            if isinstance(turn, dict) and isinstance(turn.get("value"), str):
                return turn["value"].strip()
    raise KeyError(f"id={record_id(record)} has no report text")


def reasoning_text(record: dict[str, Any]) -> str:
    for key in ("reasoning", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"id={record_id(record)} has no reasoning text")


def study_id(image: str) -> str:
    for part in Path(image).parts:
        if part.startswith("s") and part[1:].isdigit():
            return part
    raise ValueError(f"cannot infer study id from {image}")


def find_id(records: Iterable[dict[str, Any]], wanted: str, source: Path) -> dict[str, Any]:
    matches = [record for record in records if str(record_id(record)) == wanted]
    if len(matches) != 1:
        raise ValueError(f"expected one id={wanted} record in {source}, found {len(matches)}")
    return matches[0]


def find_current_image(
    records: Iterable[dict[str, Any]], current_image: str, source: Path
) -> dict[str, Any]:
    matches = []
    for record in records:
        raw = record.get("mimic_image_file_path", record.get("image"))
        if isinstance(raw, str) and Path(raw).as_posix().endswith(current_image):
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            f"expected one anatomy row for {current_image} in {source}, found {len(matches)}"
        )
    return matches[0]


def find_prior_report(
    records: Iterable[dict[str, Any]], prior_image: str, source: Path
) -> dict[str, Any]:
    matches = [record for record in records if images(record)[0] == prior_image]
    if not matches:
        raise ValueError(f"no report for prior image {prior_image} in {source}")
    reports = {report_text(record) for record in matches}
    if len(reports) != 1:
        raise ValueError(f"conflicting reports for prior image {prior_image} in {source}")
    return matches[0]


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def copy_images(output: Path, image_root: Path, relative_images: Iterable[str]) -> None:
    destination_root = output / "inputs" / "images"
    for relative in dict.fromkeys(relative_images):
        source = (image_root / relative).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        shutil.copy2(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--report-corpus", type=Path, required=True)
    parser.add_argument("--clinical-context", type=Path, required=True)
    parser.add_argument("--few-shot-examples", type=Path, required=True)
    parser.add_argument("--anatomy-source", type=Path, required=True)
    parser.add_argument("--reasoning-predictions", type=Path, required=True)
    parser.add_argument("--bbox-predictions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--few-shot-count", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if args.few_shot_count <= 0:
        raise ValueError("--few-shot-count must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to populate non-empty directory: {output}")

    report_records = load_records(args.report_corpus)
    base_source = find_id(report_records, args.record_id, args.report_corpus)
    base_images = images(base_source)
    if len(base_images) < 2 or base_images[0] == base_images[1]:
        raise ValueError("selected record must contain distinct current and prior images")
    current_image, prior_image = base_images[:2]
    prior_source = find_prior_report(report_records, prior_image, args.report_corpus)
    base_record = {
        "id": record_id(base_source),
        "image": base_images,
        "report": report_text(base_source),
    }
    prior_record = {
        "id": record_id(prior_source),
        "image": [prior_image],
        "findings": report_text(prior_source),
    }
    del report_records

    context_source = load_object(args.clinical_context)
    current_study = study_id(current_image)
    context = context_source.get(current_study)
    if not isinstance(context, dict):
        raise KeyError(f"clinical context is missing {current_study}")
    del context_source

    few_shot = load_records(args.few_shot_examples)[: args.few_shot_count]
    if len(few_shot) != args.few_shot_count:
        raise ValueError(
            f"requested {args.few_shot_count} few-shot records but found {len(few_shot)}"
        )

    anatomy = dict(
        find_current_image(
            load_records(args.anatomy_source), current_image, args.anatomy_source
        )
    )
    # Keep the real row and pixel-space boxes, but make its image path portable
    # within the extracted sample instead of retaining a machine-specific path.
    anatomy["mimic_image_file_path"] = current_image
    reasoning = find_id(
        load_records(args.reasoning_predictions), args.record_id, args.reasoning_predictions
    )
    bbox = find_id(load_records(args.bbox_predictions), args.record_id, args.bbox_predictions)

    write_json(output / "inputs" / "base_records.json", [base_record])
    write_json(output / "inputs" / "reference_reports.json", [prior_record])
    write_json(output / "inputs" / "clinical_context.json", {current_study: context})
    write_json(output / "inputs" / "few_shot_examples.json", few_shot)
    write_json(output / "inputs" / "anatomy_boxes.json", [anatomy])
    write_json(
        output / "gpt_outputs" / "reasoning_predictions.json",
        [{"id": record_id(reasoning), "reasoning": reasoning_text(reasoning)}],
    )
    write_json(
        output / "gpt_outputs" / "bbox_predictions.json",
        [{"id": record_id(bbox), "reasoning": reasoning_text(bbox)}],
    )
    copy_images(output, args.image_root.resolve(), base_images)
    # The historical project bbox predictions were generated with x/height and
    # y/width scaling. The replay script reads this marker and reproduces those
    # coordinates exactly; new datasets should use the standard convention.
    (output / ".legacy_swapped_dimensions").write_text("", encoding="utf-8")

    readme = output / "README.md"
    readme.write_text(
        "# Private real-data sample\n\n"
        f"This ignored directory contains local restricted data for record `{args.record_id}`. "
        "Do not commit or redistribute it. The JSON inputs, current/prior images, and "
        "both saved generation stages were extracted from the real local project data.\n\n"
        "This historical sample uses the legacy swapped-dimension coordinate convention; "
        "the marker in this directory enables it automatically during replay.\n\n"
        "Replay the deterministic stages from the repository root with:\n\n"
        "```bash\n"
        "MCOT_SAMPLE_ROOT=\"$PWD/data/sample\" bash scripts/run_sample_data_pipeline.sh\n"
        "```\n",
        encoding="utf-8",
    )
    print(f"wrote private real-data sample id={args.record_id} to {output}")
    print(f"copied {len(base_images)} images from {args.image_root.resolve()}")


if __name__ == "__main__":
    main()
