"""Join report metadata with one or more id-keyed reasoning result shards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from project_paths import BASE_RECORDS, REASONING, REASONING_SHARD

from .common import dump_json, image_list, load_records, reasoning_text, record_id, report_text


def merge_reasoning(
    base_records: list[dict[str, Any]],
    result_shards: list[list[dict[str, Any]]],
    *,
    allow_overwrite: bool,
    allow_missing: bool,
) -> list[dict[str, Any]]:
    generated: dict[str, str] = {}
    for shard in result_shards:
        for record in shard:
            key = str(record_id(record))
            value = reasoning_text(record)
            if key in generated and generated[key] != value and not allow_overwrite:
                raise ValueError(
                    f"conflicting reasoning for id={key}; use --allow-overwrite for correction shards"
                )
            generated[key] = value

    merged: list[dict[str, Any]] = []
    missing: list[Any] = []
    base_ids: set[str] = set()
    for record in base_records:
        item_id = record_id(record)
        key = str(item_id)
        if key in base_ids:
            raise ValueError(f"duplicate id in base records: {item_id}")
        base_ids.add(key)
        value = generated.get(key)
        if value is None:
            missing.append(item_id)
            continue
        merged.append(
            {
                "id": item_id,
                "reasoning": value,
                "report": report_text(record),
                "image": image_list(record),
            }
        )

    extras = sorted(set(generated) - base_ids)
    if extras:
        raise ValueError(f"reasoning results contain {len(extras)} ids absent from the base corpus")
    if missing and not allow_missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(f"missing reasoning for {len(missing)} base records (first ids: {preview})")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=BASE_RECORDS, help="JSON/JSONL report corpus."
    )
    parser.add_argument(
        "--reasoning",
        type=Path,
        nargs="+",
        default=[REASONING_SHARD],
        help="One or more id/reasoning JSON shards, in precedence order.",
    )
    parser.add_argument("--output", type=Path, default=REASONING)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = merge_reasoning(
        load_records(args.base),
        [load_records(path) for path in args.reasoning],
        allow_overwrite=args.allow_overwrite,
        allow_missing=args.allow_missing,
    )
    dump_json(args.output, merged, overwrite=args.overwrite)
    print(f"wrote {len(merged)} records to {args.output}")


if __name__ == "__main__":
    main()
