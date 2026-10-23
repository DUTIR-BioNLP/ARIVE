"""Attach normalized anatomy boxes to refined records by current-image path."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from project_paths import (
    MISSING_ANATOMY,
    NORMALIZED_ANATOMY,
    REASONING,
    REASONING_WITH_BOXES,
)

from .common import dump_json, image_list, load_records


def image_key(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def attach_boxes(
    records: list[dict[str, Any]], anatomy: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anatomy_by_image: dict[str, list[dict[str, Any]]] = {}
    for item in anatomy:
        raw_image = item.get("image", item.get("mimic_image_file_path"))
        diseases = item.get("disease")
        if not isinstance(raw_image, str) or not isinstance(diseases, list):
            raise ValueError("each anatomy record needs image and disease fields")
        key = image_key(raw_image)
        if key in anatomy_by_image:
            raise ValueError(f"duplicate anatomy image: {raw_image}")
        anatomy_by_image[key] = diseases

    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for record in records:
        current_image = image_list(record)[0]
        diseases = anatomy_by_image.get(image_key(current_image))
        if diseases is None:
            missing.append(record)
            continue
        item = dict(record)
        item["disease"] = diseases
        matched.append(item)
    return matched, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=REASONING)
    parser.add_argument("--anatomy", type=Path, nargs="+", default=[NORMALIZED_ANATOMY])
    parser.add_argument("--output", type=Path, default=REASONING_WITH_BOXES)
    parser.add_argument(
        "--missing-output",
        type=Path,
        default=MISSING_ANATOMY,
        help="Optional path for records without an anatomy match.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anatomy = []
    for path in args.anatomy:
        anatomy.extend(load_records(path))
    matched, missing = attach_boxes(load_records(args.records), anatomy)
    dump_json(args.output, matched, overwrite=args.overwrite)
    if args.missing_output is not None:
        dump_json(args.missing_output, missing, overwrite=args.overwrite)
    print(
        f"wrote {len(matched)} bbox-enriched records to {args.output}; "
        f"missing anatomy={len(missing)}"
    )


if __name__ == "__main__":
    main()
