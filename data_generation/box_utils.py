"""Parsing and validation helpers for ARIVE's normalized bounding-box tags."""

from __future__ import annotations

import re


BOX_RE = re.compile(
    r"<\|box_start\|>\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*<\|box_end\|>"
)
OBJECT_REF_BOX_RE = re.compile(
    r"<\|object_ref_start\|>.*?<\|object_ref_end\|>\s*"
    + BOX_RE.pattern,
    re.DOTALL,
)


def format_box(coords: tuple[int, int, int, int]) -> str:
    return f"<|box_start|>{coords[0]} {coords[1]} {coords[2]} {coords[3]}<|box_end|>"


def coordinates(text: str) -> list[tuple[int, int, int, int]]:
    return [tuple(int(match.group(i)) for i in range(1, 5)) for match in BOX_RE.finditer(text)]


def _remove_unmatched_end_tokens(text: str) -> str:
    start_token = "<|box_start|>"
    end_token = "<|box_end|>"
    cursor = 0
    depth = 0
    pieces: list[str] = []
    while cursor < len(text):
        start = text.find(start_token, cursor)
        end = text.find(end_token, cursor)
        if start == -1 and end == -1:
            pieces.append(text[cursor:])
            break
        if start != -1 and (end == -1 or start < end):
            pieces.append(text[cursor : start + len(start_token)])
            cursor = start + len(start_token)
            depth += 1
        else:
            pieces.append(text[cursor:end])
            cursor = end + len(end_token)
            if depth > 0:
                pieces.append(end_token)
                depth -= 1
    return "".join(pieces)


def normalize_reasoning_boxes(text: str) -> str:
    """Repair the known end-token typo, remove stray ends, and deduplicate boxes."""
    text = re.sub(r"<\|box_end>(?!\|)", "<|box_end|>", text)
    text = _remove_unmatched_end_tokens(text)
    seen: set[tuple[int, int, int, int]] = set()

    def replace(match: re.Match[str]) -> str:
        coords = tuple(int(match.group(i)) for i in range(1, 5))
        if coords in seen:
            return ""
        seen.add(coords)
        return format_box(coords)

    normalized = BOX_RE.sub(replace, text)
    residue = BOX_RE.sub("", normalized)
    if "<|box_start|>" in residue or "<|box_end|>" in residue:
        raise ValueError("reasoning contains an incomplete or malformed bounding-box tag")
    return normalized


def validate_normalized_coordinates(text: str) -> None:
    for coords in coordinates(text):
        x1, y1, x2, y2 = coords
        if any(value < 0 or value > 1000 for value in coords):
            raise ValueError(f"bounding box is outside normalized [0, 1000] coordinates: {coords}")
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"bounding box has non-positive width or height: {coords}")

