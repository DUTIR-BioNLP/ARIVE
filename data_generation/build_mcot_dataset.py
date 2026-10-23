"""Merge bbox-insertion shards and validate the resulting ARIVE reasoning."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
import re

from project_paths import BBOX_SHARD, MCOT, REASONING_WITH_BOXES

from .box_utils import BOX_RE, coordinates, normalize_reasoning_boxes, validate_normalized_coordinates
from .common import dump_json, image_list, load_records, reasoning_text, record_id, report_text


CONCLUSION_START = re.compile(r"(?i)^(?:in summary|in conclusion|overall)\b")


def _named_lung_zone_sentence(source: str, label: str) -> int | None:
    """Locate an explicitly named lung zone; leave implied regions to the model."""
    region = re.fullmatch(r"(right|left) (upper|mid|lower) lung zone", label)
    if not region:
        return None
    side, zone = region.groups()
    separator = r"[\s\-‑–]+"
    location = zone + separator + r"(?:lung|lobe|zone|field)"
    if zone == "mid":
        location = r"(?:mid|middle)" + separator + r"(?:lung|lobe|zone|field)"
    elif zone == "lower":
        location = rf"(?:{location}|(?:lung{separator})?base)"
    match = re.search(rf"\b{side}{separator}{location}\b", source, re.I)
    if not match:
        return None
    start = 0
    for boundary in re.finditer(r"(?<=[.!?])\s+|\n\s*\n", source):
        if boundary.end() > match.start():
            break
        start = boundary.end()
    return start


def preserve_narrative_with_boxes(
    source: str, predicted: str, anatomy: list[dict[str, Any]] | None = None,
    *, reanchor_named_regions: bool = True,
) -> str:
    """Transfer inserted boxes to source offsets without accepting text edits.

    Whitespace is ignored for alignment, while the original source characters
    are retained. Reject heavily rewritten text or ambiguous anchors instead
    of assigning a box to an unrelated observation.
    """
    plain = BOX_RE.sub("", predicted)
    source_offsets = [i for i, char in enumerate(source) if not char.isspace()]
    predicted_offsets = [i for i, char in enumerate(plain) if not char.isspace()]
    source_key = "".join(source[i] for i in source_offsets)
    predicted_key = "".join(plain[i] for i in predicted_offsets)
    matcher = SequenceMatcher(None, predicted_key, source_key, autojunk=False)
    blocks = matcher.get_matching_blocks()
    # Missing source sentences can be restored safely; new or rewritten
    # predicted content is what makes a box placement unreliable.
    retained_fraction = sum(block.size for block in blocks) / max(len(predicted_key), 1)
    if retained_fraction < 0.8:
        raise ValueError("bbox pass rewrote too much text to align safely")
    labels_by_box: dict[tuple[int, ...], list[str]] = {}
    for entry in anatomy or []:
        labels_by_box.setdefault(tuple(entry["bbox_coordinates"]), []).append(entry["bbox_labels"])
    placements: list[tuple[int, int, str]] = []
    removed = 0
    for match in BOX_RE.finditer(predicted):
        plain_offset = match.start() - removed
        anchor = bisect_left(predicted_offsets, plain_offset)
        block = next((b for b in blocks if b.a <= anchor < b.a + b.size), None)
        if (block is None
                or block.a + block.size - anchor < min(6, len(predicted_key) - anchor)):
            raise ValueError("bbox anchor lies in rewritten or unsupported text")
        source_offset = source_offsets[block.b + anchor - block.a]
        if (source_offset and source[source_offset - 1].isalnum()
                and source[source_offset].isalnum()):
            raise ValueError("bbox anchor would split a source word")
        labels = labels_by_box.get(tuple(map(int, match.groups())), [])
        named_offset = (_named_lung_zone_sentence(source, labels[0])
                        if reanchor_named_regions and len(labels) == 1 else None)
        placements.append((source_offset, named_offset if named_offset is not None else source_offset,
                           match.group()))
        removed += match.end() - match.start()
    insertions: dict[int, list[str]] = {}
    proposed_offsets = [proposed for _, proposed, _ in placements]
    for original, proposed, tag in placements:
        # Keep the model anchor when relocating would create adjacent boxes.
        offset = proposed if proposed_offsets.count(proposed) == 1 else original
        insertions.setdefault(offset, []).append(tag)
    result, cursor = [], 0
    for offset, tags in sorted(insertions.items()):
        result.extend([source[cursor:offset], " ".join(tags), "\n"])
        cursor = offset
    result.append(source[cursor:])
    return "".join(result)


def restore_missing_conclusion(source_reasoning: str, predicted: str) -> str:
    """Restore a final conclusion silently omitted by the bbox-only model pass."""
    source_paragraphs = [part.strip() for part in re.split(r"\n\s*\n", source_reasoning) if part.strip()]
    predicted_paragraphs = [part.strip() for part in re.split(r"\n\s*\n", predicted) if part.strip()]
    if not source_paragraphs or not predicted_paragraphs:
        return predicted
    conclusion = source_paragraphs[-1]
    if not CONCLUSION_START.match(conclusion):
        return predicted
    # The model sometimes splits the conclusion across two paragraphs, or
    # inserts a box before a final continuation paragraph.  Any retained
    # conclusion cue means appending the source paragraph would duplicate it.
    clean_predicted = re.sub(r"<\|[^>]+\|>|<image>", "", predicted)
    if (re.search(r"(?i)\b(?:in summary|in conclusion)\b", clean_predicted)
            or re.search(r"(?im)(?:^|\n)\s*overall\b", clean_predicted)):
        return predicted
    if conclusion in predicted:
        return predicted
    return predicted.rstrip() + "\n\n" + conclusion


def build_mcot_dataset(
    bbox_records: list[dict[str, Any]],
    prediction_shards: list[list[dict[str, Any]]],
    *,
    allow_overwrite: bool,
    allow_missing: bool,
    validate_coordinates: bool,
    restore_conclusion: bool = False,
    preserve_source_text: bool = False,
) -> list[dict[str, Any]]:
    predictions: dict[str, str] = {}
    anchor_verified: dict[str, bool] = {}
    for shard in prediction_shards:
        for record in shard:
            key = str(record_id(record))
            value = reasoning_text(record)
            if key in predictions and predictions[key] != value and not allow_overwrite:
                raise ValueError(
                    f"conflicting bbox reasoning for id={key}; use --allow-overwrite for corrections"
                )
            predictions[key] = value
            anchor_verified[key] = record.get("grounding_method") == "source_sentence_anchors"

    output: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    missing: list[Any] = []
    for source in bbox_records:
        item_id = record_id(source)
        key = str(item_id)
        if key in source_ids:
            raise ValueError(f"duplicate id in bbox source: {item_id}")
        source_ids.add(key)
        predicted = predictions.get(key)
        if predicted is None:
            missing.append(item_id)
            continue
        predicted = normalize_reasoning_boxes(predicted)
        if preserve_source_text:
            allowed = {tuple(d["bbox_coordinates"]) for d in source.get("disease", [])}
            if any(box not in allowed for box in coordinates(predicted)):
                raise ValueError(f"id={item_id} has a box absent from the supplied anatomy")
            predicted = preserve_narrative_with_boxes(
                reasoning_text(source), predicted, source.get("disease", []),
                reanchor_named_regions=not anchor_verified[key],
            )
        elif restore_conclusion:
            predicted = restore_missing_conclusion(reasoning_text(source), predicted)
        if validate_coordinates:
            validate_normalized_coordinates(predicted)
        output.append(
            {
                "id": item_id,
                "reasoning": predicted,
                "report": report_text(source),
                "image": image_list(source),
            }
        )

    extras = sorted(set(predictions) - source_ids)
    if extras:
        raise ValueError(f"bbox predictions contain {len(extras)} ids absent from the source")
    if missing and not allow_missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(f"missing bbox reasoning for {len(missing)} records (first ids: {preview})")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preserve-source-text", action="store_true",
        help="Transfer box placements to the original narrative, rejecting unsafe alignment.",
    )
    parser.add_argument(
        "--bbox-source",
        type=Path,
        default=REASONING_WITH_BOXES,
        help="JSON with id/report/image/disease used by the bbox insertion stage.",
    )
    parser.add_argument("--predictions", type=Path, nargs="+", default=[BBOX_SHARD])
    parser.add_argument("--output", type=Path, default=MCOT)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--validate-coordinates",
        action="store_true",
        help="Require every generated coordinate to lie in [0,1000] with positive area.",
    )
    parser.add_argument(
        "--restore-missing-conclusion", action="store_true",
        help="Restore a final summary paragraph dropped by bbox insertion.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_mcot_dataset(
        load_records(args.bbox_source),
        [load_records(path) for path in args.predictions],
        allow_overwrite=args.allow_overwrite,
        allow_missing=args.allow_missing,
        validate_coordinates=args.validate_coordinates,
        restore_conclusion=args.restore_missing_conclusion,
        preserve_source_text=args.preserve_source_text,
    )
    dump_json(args.output, data, overwrite=args.overwrite)
    print(f"wrote {len(data)} records to {args.output}")


if __name__ == "__main__":
    main()
