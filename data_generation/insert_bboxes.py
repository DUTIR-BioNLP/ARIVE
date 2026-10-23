"""Spatial pass: insert relevant normalized anatomy boxes into ARIVE reasoning."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from project_paths import BBOX_SHARD, REASONING_WITH_BOXES

from .common import add_generation_arguments, load_records, run_harmony_stage


def anatomical_information(diseases: Any) -> str:
    if not isinstance(diseases, list):
        raise TypeError("record['disease'] must be a list")
    lines: list[str] = []
    for disease in diseases:
        if not isinstance(disease, dict):
            raise TypeError("each disease entry must be an object")
        lines.append(
            json.dumps(
                {
                    "Anatomical region": disease.get("bbox_labels"),
                    "Bounding box": disease.get("bbox_coordinates"),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def build_prompt(record: dict[str, Any]) -> str:
    reasoning = record.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise KeyError("bbox record has no non-empty reasoning")
    regions = region_entries(record)
    description = "\n".join(f"{i}: {entry['bbox_labels']}" for i, entry in enumerate(regions))
    spans = sentence_spans(reasoning)
    sentences = "\n".join(f"{i}: {reasoning[start:end].strip()}"
                          for i, (start, end) in enumerate(spans))
    return f"""Select anatomical regions for the numbered observation sentences. Return only a JSON array of {{"region_id": integer, "sentence_id": integer}}. Code inserts the source box immediately before that sentence; do not copy or rewrite the narrative.

Choose only supplied regions discussed by the sentence and prefer the most localizing region. Use each region once and at most one region per sentence. Ground actual observations, not hypothetical diagnoses, clinical history, or the final summary. Separate right/left and upper/mid/lower regions when their observations are in distinct sentences. These are anatomical regions, not measured lesion boxes.

The regional finding must be supported by the current report below. Choose a broader supplied region when the exact anatomy is absent; nearby structures are not interchangeable.

<current_report>
{record.get('report', record.get('findings', ''))}
</current_report>

<anatomy>
{description}
</anatomy>
<sentences>
{sentences}
</sentences>

Return only the JSON array."""


def region_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Give coordinate aliases one stable region ID."""
    entries = []
    seen = set()
    for entry in record["disease"]:
        box = tuple(entry["bbox_coordinates"])
        if box not in seen:
            entries.append(entry)
            seen.add(box)
    return entries


def sentence_spans(narrative: str) -> list[tuple[int, int]]:
    """Keep exact source offsets, including decimal numbers and quoted endings."""
    boundaries = re.finditer(
        r"(?<=[.!?])[\"”’')\]]*\s+(?=[A-Z])|\n[ \t]*\n", narrative,
    )
    spans, start = [], 0
    for boundary in boundaries:
        end = boundary.start()
        while end < boundary.end() and not narrative[end].isspace():
            end += 1
        if narrative[start:end].strip():
            spans.append((start, end))
        start = boundary.end()
    if narrative[start:].strip():
        spans.append((start, len(narrative)))
    return spans


def insert_selected_ids(record: dict[str, Any], response: str) -> str:
    """Insert only existing regions at existing sentence starts, preserving text."""
    response = response.strip()
    if response.startswith("```json\n") and response.endswith("\n```"):
        response = response[8:-4]
    selections = json.loads(response)
    if not isinstance(selections, list) or not selections:
        raise ValueError("bbox selection must be a non-empty JSON array")
    regions = region_entries(record)
    narrative = record["reasoning"]
    sentences = sentence_spans(narrative)
    placements, used = {}, set()
    for selection in selections:
        if not isinstance(selection, dict):
            raise ValueError("bbox selection entries must be objects")
        region, sentence = selection.get("region_id"), selection.get("sentence_id")
        if (type(region) is not int or not 0 <= region < len(regions)
                or type(sentence) is not int or not 0 <= sentence < len(sentences)):
            raise ValueError(f"invalid region or sentence ID: {selection}")
        if region in used:
            raise ValueError("duplicate region ID")
        if sentence in placements:
            raise ValueError("multiple regions selected at one sentence")
        used.add(region)
        box = regions[region]["bbox_coordinates"]
        placements[sentence] = f"<|box_start|>{' '.join(map(str, box))}<|box_end|>\n"
    result = narrative
    for sentence, tag in sorted(placements.items(), reverse=True):
        offset = sentences[sentence][0]
        result = result[:offset] + tag + result[offset:]
    return result


def insert_selected_anchors(record: dict[str, Any], response: str) -> str:
    """Insert validated source boxes at exact source quotes; never accept rewrites."""
    response = response.strip()
    if response.startswith("```json\n") and response.endswith("\n```"):
        response = response[8:-4]
    selections = json.loads(response)
    if not isinstance(selections, list) or not selections:
        raise ValueError("bbox selection must be a non-empty JSON array")
    narrative = record["reasoning"]
    regions: dict[str, list[int]] = {}
    for entry in record["disease"]:
        label, box = entry["bbox_labels"], entry["bbox_coordinates"]
        if label in regions and regions[label] != box:
            raise ValueError(f"ambiguous supplied region: {label}")
        regions[label] = box
    placements: dict[int, str] = {}
    used_boxes: set[tuple[int, ...]] = set()
    for selection in selections:
        if not isinstance(selection, dict):
            raise ValueError("bbox selection entries must be objects")
        label, quote = selection.get("region"), selection.get("before")
        if not isinstance(label, str) or label not in regions:
            raise ValueError(f"bbox region absent from supplied anatomy: {label}")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("bbox insertion anchor must be a non-empty source quote")
        expression = r"\s+".join(re.escape(word) for word in quote.split())
        matches = list(re.finditer(expression + r"(?!\w)", narrative))
        if not matches and len(quote.split()) >= 8:
            words = quote.split()
            # A copied quote can contain a typo at its tail. A unique literal
            # prefix still anchors the original sentence without editing text.
            for drop in range(1, min(5, len(words) - 6) + 1):
                expression = r"\s+".join(re.escape(word) for word in words[:-drop])
                candidates = list(re.finditer(expression + r"(?!\w)", narrative))
                if len(candidates) == 1:
                    matches = candidates
                    break
        if len(matches) != 1:
            raise ValueError(f"bbox anchor must match source exactly once: {quote!r}")
        match = matches[0]
        prefix = narrative[:match.start()].rstrip().rstrip('"\'”’)]}')
        if prefix and prefix[-1] not in ".!?" and not narrative[:match.start()].endswith("\n\n"):
            raise ValueError(f"bbox anchor must begin a sentence: {quote!r}")
        if (match.end() < len(narrative) and narrative[match.end()].isalnum()
                and narrative[match.end() - 1].isalnum()):
            raise ValueError("bbox quote ends inside a source word")
        box = tuple(regions[label])
        if box in used_boxes:
            raise ValueError("duplicate bbox")
        # A sentence can mention several central structures. Keep the first
        # selected region rather than creating a crop with no following text.
        if match.start() in placements:
            continue
        used_boxes.add(box)
        placements[match.start()] = f"<|box_start|>{' '.join(map(str, box))}<|box_end|>\n"
    result = narrative
    for offset, tag in sorted(placements.items(), reverse=True):
        result = result[:offset] + tag + result[offset:]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REASONING_WITH_BOXES,
        help="BBox-enriched JSON records with id/report/reasoning/disease.",
    )
    add_generation_arguments(parser, default_output=BBOX_SHARD)
    parser.set_defaults(max_model_len=30720, max_tokens=30720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_harmony_stage(
        records=load_records(args.input), prompt_builder=build_prompt, args=args,
        developer_instruction=(
            "You select grounding regions for an existing annotation. "
            "Return only a JSON array of supplied region IDs and sentence IDs. "
            "Never rewrite the clinical narrative or invent a region."
        ),
        record_output_transform=insert_selected_ids,
        raw_output_path=args.output.parent / "raw" / args.output.name,
        output_fields={"grounding_method": "source_sentence_anchors"},
    )


if __name__ == "__main__":
    main()
