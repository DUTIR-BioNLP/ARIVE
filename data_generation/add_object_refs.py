#!/usr/bin/env python3
"""Add object reference tags before matched bbox tags in ARIVE reasoning."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from project_paths import MCOT, MCOT_WITH_OBJECT_REFS, REASONING_WITH_BOXES


BOX_RE = re.compile(
    r"<\|box_start\|>\s*"
    r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)"
    r"\s*<\|box_end\|>"
)
OBJECT_REF_BEFORE_RE = re.compile(
    r"<\|object_ref_start\|>.*?<\|object_ref_end\|>\s*$",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match bbox tags in MCOT_dataset.json reasoning against "
            "bbx_dataset.json disease bbox_coordinates, then add object_ref tags."
        )
    )
    parser.add_argument(
        "--mcot", type=Path, default=MCOT, help="Input grounded reasoning JSON."
    )
    parser.add_argument(
        "--bbx", type=Path, default=REASONING_WITH_BOXES, help="Input bbox JSON."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MCOT_WITH_OBJECT_REFS,
        help="Output updated reasoning JSON.",
    )
    parser.add_argument(
        "--join-labels-with",
        default="; ",
        help="Separator used when one coordinate has multiple bbox_labels.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of pretty printed JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any, *, compact: bool, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if compact:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")


def normalize_coords(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(int(v) for v in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def format_box(coords: tuple[int, int, int, int]) -> str:
    return f"<|box_start|>{coords[0]} {coords[1]} {coords[2]} {coords[3]}<|box_end|>"


def build_bbx_by_id(bbx_data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    bbx_by_id: dict[str, list[dict[str, Any]]] = {}
    for item in bbx_data:
        if "id" not in item:
            raise KeyError("each bbox record must contain an id")
        coord_to_labels: dict[tuple[int, int, int, int], list[str]] = {}
        for disease in item.get("disease") or []:
            coords = normalize_coords(disease.get("bbox_coordinates"))
            label = disease.get("bbox_labels")
            if coords is None or label is None:
                continue
            label = str(label)
            labels = coord_to_labels.setdefault(coords, [])
            if label not in labels:
                labels.append(label)
        key = str(item["id"])
        if key in bbx_by_id:
            raise ValueError(f"duplicate bbox id: {item['id']}")
        bbx_by_id[key] = [
            {"bbox_coordinates": coords, "bbox_labels": labels}
            for coords, labels in coord_to_labels.items()
        ]
    return bbx_by_id


def get_context_window(reasoning: str, match: re.Match[str]) -> str:
    next_box = BOX_RE.search(reasoning, match.end())
    after_end = next_box.start() if next_box else min(len(reasoning), match.end() + 300)
    before = reasoning[max(0, match.start() - 120) : match.start()]
    after = reasoning[match.end() : after_end]
    return f"{before} {after}".lower()


def label_context_score(labels: list[str], context: str) -> int:
    score = 0
    for label in labels:
        normalized_label = label.lower()
        if normalized_label in context:
            score += 1000
        for token in normalized_label.replace("-", " ").split():
            if len(token) >= 3 and token in context:
                score += 120
    return score


def coord_similarity_score(
    box_coords: tuple[int, int, int, int],
    candidate_coords: tuple[int, int, int, int],
) -> tuple[int, int, int, int, int, float, float]:
    positional_exact = sum(1 for a, b in zip(box_coords, candidate_coords) if a == b)
    positional_close_10 = sum(1 for a, b in zip(box_coords, candidate_coords) if abs(a - b) <= 10)
    positional_close_50 = sum(1 for a, b in zip(box_coords, candidate_coords) if abs(a - b) <= 50)

    # Count similar numbers even if one malformed coordinate moved to another slot.
    unmatched_candidate = list(candidate_coords)
    any_position_close_60 = 0
    for value in box_coords:
        best_index = None
        best_delta = None
        for index, candidate_value in enumerate(unmatched_candidate):
            delta = abs(value - candidate_value)
            if delta <= 60 and (best_delta is None or delta < best_delta):
                best_index = index
                best_delta = delta
        if best_index is not None:
            any_position_close_60 += 1
            unmatched_candidate.pop(best_index)

    total_abs_delta = sum(abs(a - b) for a, b in zip(box_coords, candidate_coords))
    center_delta = abs((box_coords[0] + box_coords[2]) - (candidate_coords[0] + candidate_coords[2]))
    center_delta += abs((box_coords[1] + box_coords[3]) - (candidate_coords[1] + candidate_coords[3]))
    box_area = max(0, box_coords[2] - box_coords[0]) * max(0, box_coords[3] - box_coords[1])
    candidate_area = max(0, candidate_coords[2] - candidate_coords[0]) * max(
        0, candidate_coords[3] - candidate_coords[1]
    )
    area_delta = abs(box_area - candidate_area)
    return (
        positional_exact,
        positional_close_10,
        positional_close_50,
        any_position_close_60,
        -total_abs_delta,
        -center_delta,
        -area_delta,
    )


def find_best_candidate(
    box_coords: tuple[int, int, int, int],
    candidates: list[dict[str, Any]],
    reasoning: str,
    match: re.Match[str],
) -> dict[str, Any] | None:
    if not candidates:
        return None

    context = get_context_window(reasoning, match)
    best_candidate = None
    best_score = None
    for candidate in candidates:
        candidate_coords = candidate["bbox_coordinates"]
        labels = candidate["bbox_labels"]
        score = (
            label_context_score(labels, context),
            *coord_similarity_score(box_coords, candidate_coords),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate


def add_object_refs(
    mcot_data: list[dict[str, Any]],
    bbx_by_id: dict[str, list[dict[str, Any]]],
    *,
    join_labels_with: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    stats: Counter[str] = Counter()

    for index, item in enumerate(mcot_data):
        item_id = item.get("id")
        candidates = bbx_by_id.get(str(item_id))
        if candidates is None:
            stats["missing_bbx_id"] += 1
            candidates = []
        coord_to_labels = {
            candidate["bbox_coordinates"]: candidate["bbox_labels"]
            for candidate in candidates
        }

        reasoning = item.get("reasoning")
        if not isinstance(reasoning, str):
            stats["missing_reasoning"] += 1
            continue

        def replace(match: re.Match[str]) -> str:
            stats["box_tags_seen"] += 1
            coords = tuple(int(match.group(i)) for i in range(1, 5))
            normalized_box = format_box(coords)
            before = reasoning[: match.start()]
            if OBJECT_REF_BEFORE_RE.search(before):
                stats["already_tagged"] += 1
                return normalized_box

            labels = coord_to_labels.get(coords)
            if not labels:
                candidate = find_best_candidate(coords, candidates, reasoning, match)
                if candidate is None:
                    stats["unmatched_boxes"] += 1
                    return normalized_box
                labels = candidate["bbox_labels"]
                stats["fuzzy_matched_boxes"] += 1

            if len(labels) > 1:
                stats["ambiguous_coordinate_matches"] += 1
            label_text = join_labels_with.join(labels)
            stats["matched_boxes"] += 1
            return f"<|object_ref_start|>{label_text}<|object_ref_end|>{normalized_box}"

        item["reasoning"] = BOX_RE.sub(replace, reasoning)

    return mcot_data, stats


def main() -> None:
    args = parse_args()
    mcot_data = load_json(args.mcot)
    bbx_data = load_json(args.bbx)

    if not isinstance(mcot_data, list):
        raise TypeError(f"{args.mcot} must contain a JSON list")
    if not isinstance(bbx_data, list):
        raise TypeError(f"{args.bbx} must contain a JSON list")

    bbx_by_id = build_bbx_by_id(bbx_data)
    updated, stats = add_object_refs(
        mcot_data,
        bbx_by_id,
        join_labels_with=args.join_labels_with,
    )

    dump_json(args.output, updated, compact=args.compact, overwrite=args.overwrite)

    print(f"ARIVE records: {len(mcot_data)}")
    print(f"BBX records: {len(bbx_data)}")
    print(f"Box tags seen: {stats['box_tags_seen']}")
    print(f"Matched boxes: {stats['matched_boxes']}")
    print(f"Fuzzy matched boxes: {stats['fuzzy_matched_boxes']}")
    print(f"Already tagged boxes: {stats['already_tagged']}")
    print(f"Unmatched boxes: {stats['unmatched_boxes']}")
    print(f"Records missing BBX id: {stats['missing_bbx_id']}")
    print(f"Records missing reasoning: {stats['missing_reasoning']}")
    print(f"Ambiguous coordinate matches: {stats['ambiguous_coordinate_matches']}")
    print(f"Wrote updated JSON: {args.output}")


if __name__ == "__main__":
    main()
