"""Normalize Chest ImaGenome/RGRG anatomy boxes into ARIVE's [0, 1000] format."""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import Any

from PIL import Image

from project_paths import ANATOMY_SOURCE, IMAGE_ROOT, NORMALIZED_ANATOMY

from .common import dump_json, load_records


ANATOMICAL_REGIONS = [
    "right lung",
    "right upper lung zone",
    "right mid lung zone",
    "right lower lung zone",
    "right hilar structures",
    "right apical zone",
    "right costophrenic angle",
    "right hemidiaphragm",
    "left lung",
    "left upper lung zone",
    "left mid lung zone",
    "left lower lung zone",
    "left hilar structures",
    "left apical zone",
    "left costophrenic angle",
    "left hemidiaphragm",
    "trachea",
    "spine",
    "right clavicle",
    "left clavicle",
    "aortic arch",
    "mediastinum",
    "upper mediastinum",
    "svc",
    "cardiac silhouette",
    "cavoatrial junction",
    "right atrium",
    "carina",
    "abdomen",
]
LIST_COLUMNS = {
    "bbox_coordinates",
    "bbox_labels",
    "bbox_phrases",
    "bbox_phrase_exists",
    "bbox_is_abnormal",
}


def load_source(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".csv":
        return load_records(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = dict(row)
            for key in LIST_COLUMNS:
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    parsed[key] = ast.literal_eval(value)
            rows.append(parsed)
    return rows


def normalize_box(
    box: list[Any],
    *,
    width: int,
    height: int,
    legacy_swapped_dimensions: bool,
) -> list[int]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"invalid pixel-space box: {box!r}")
    x_scale = width
    y_scale = height
    x1, y1, x2, y2 = (float(value) for value in box)
    normalized = [
        round(x1 / x_scale * 1000),
        round(y1 / y_scale * 1000),
        round(x2 / x_scale * 1000),
        round(y2 / y_scale * 1000),
    ]
    return [max(0, min(1000, value)) for value in normalized]


def region_name(value: Any) -> str:
    if isinstance(value, str) and not value.strip().isdigit():
        return value.strip()
    index = int(value) - 1
    if not 0 <= index < len(ANATOMICAL_REGIONS):
        raise ValueError(f"anatomical region index is outside 1..{len(ANATOMICAL_REGIONS)}: {value}")
    return ANATOMICAL_REGIONS[index]


def bool_at(values: Any, index: int) -> bool:
    return isinstance(values, list) and index < len(values) and bool(values[index])


def prepare_record(
    record: dict[str, Any],
    *,
    image_root: Path,
    keep_all: bool,
    legacy_swapped_dimensions: bool,
) -> dict[str, Any]:
    raw_path = record.get("mimic_image_file_path", record.get("image"))
    if not isinstance(raw_path, str):
        raise KeyError("anatomy row needs mimic_image_file_path or image")
    source_path = Path(raw_path)
    image_path = source_path if source_path.is_absolute() else image_root / source_path
    image_path = image_path.resolve()
    try:
        relative_image = image_path.relative_to(image_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"image {image_path} is outside --image-root {image_root}") from exc
    with Image.open(image_path) as image:
        width, height = image.size

    boxes = record.get("bbox_coordinates")
    labels = record.get("bbox_labels")
    if not isinstance(boxes, list) or not isinstance(labels, list) or len(boxes) != len(labels):
        raise ValueError(f"box/label arrays are missing or misaligned for {relative_image}")
    phrase_flags = record.get("bbox_phrase_exists")
    abnormal_flags = record.get("bbox_is_abnormal")
    flags_available = isinstance(phrase_flags, list) or isinstance(abnormal_flags, list)

    disease = []
    for index, (box, label) in enumerate(zip(boxes, labels)):
        if not keep_all and flags_available:
            if not (bool_at(phrase_flags, index) or bool_at(abnormal_flags, index)):
                continue
        disease.append(
            {
                "bbox_labels": region_name(label),
                "bbox_coordinates": normalize_box(
                    box,
                    width=width,
                    height=height,
                    legacy_swapped_dimensions=legacy_swapped_dimensions,
                ),
            }
        )
    output = {"image": relative_image, "disease": disease}
    for key in ("subject_id", "study_id", "image_id"):
        if key in record:
            output[key] = record[key]
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ANATOMY_SOURCE, help="Source CSV/JSON/JSONL."
    )
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=NORMALIZED_ANATOMY)
    parser.add_argument("--keep-all", action="store_true", help="Ignore phrase/abnormal flags.")
    parser.add_argument(
        "--legacy-swapped-dimensions",
        action="store_true",
        help="Reproduce the historical x/height and y/width normalization convention.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        prepare_record(
            record,
            image_root=args.image_root,
            keep_all=args.keep_all,
            legacy_swapped_dimensions=args.legacy_swapped_dimensions,
        )
        for record in load_source(args.input)
    ]
    dump_json(args.output, rows, overwrite=args.overwrite)
    print(f"wrote {len(rows)} normalized anatomy records to {args.output}")


if __name__ == "__main__":
    main()
