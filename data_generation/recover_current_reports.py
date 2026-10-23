"""Reject non-diagnostic current reports or recover them before GPT generation.

MIMIC-CXR occasionally has a truncated FINDINGS field despite a complete
IMPRESSION.  The construction pipeline must not let GPT fill that gap from a
prior report.  Explicit overrides take precedence for ambiguous impressions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import zipfile

from project_paths import BASE_RECORDS

from .common import dump_json, image_list, load_records, record_id, report_text


NON_DIAGNOSTIC = re.compile(
    r"(?i)^\s*(?:send the|at the|see below|"
    r"portable\s+.*?\b(?:at\s+\d{1,2}:\d{2}|is submitted)\.?\s*)$"
)
SECTION = re.compile(r"(?ms)^\s*IMPRESSION\s*:\s*(.*?)(?=^\s*[A-Z][A-Z ]+\s*:|\Z)")


def looks_incomplete(report: str) -> bool:
    """Conservative detector for known truncated/acquisition-only reports."""
    return not report.strip() or bool(NON_DIAGNOSTIC.fullmatch(report.strip()))


def load_overrides(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("report overrides must be a JSON object mapping record IDs to reports")
    overrides = {str(key): report for key, report in value.items()}
    if any(not isinstance(report, str) or looks_incomplete(report) for report in overrides.values()):
        raise ValueError("every report override must contain a substantive report")
    return overrides


def impression_for_image(archive: zipfile.ZipFile, image: str) -> str:
    report_path = str(Path(image).parent.with_suffix(".txt"))
    try:
        raw = archive.read(report_path).decode("utf-8")
    except KeyError as exc:
        raise FileNotFoundError(f"report {report_path} missing from archive") from exc
    match = SECTION.search(raw)
    if not match or looks_incomplete(match.group(1)):
        raise ValueError(f"no usable IMPRESSION in {report_path}")
    return match.group(1).strip()


def recover_records(
    records: list[dict], *, archive: zipfile.ZipFile | None,
    overrides: dict[str, str], start: int = 0, end: int | None = None,
) -> tuple[list[dict], list[dict]]:
    output = [dict(record) for record in records]
    recovered = []
    selected_end = min(end if end is not None else len(output), len(output))
    selected_ids = {str(record_id(record)) for record in output[start:selected_end]}
    unknown = set(overrides) - selected_ids
    if unknown:
        raise ValueError(f"report override IDs are not in the selected records: {sorted(unknown)}")
    for index in range(start, selected_end):
        record = output[index]
        identifier = str(record_id(record))
        original = report_text(record)
        replacement = overrides.get(identifier)
        source = "override" if replacement is not None else None
        if replacement is None and looks_incomplete(original):
            if archive is None:
                raise ValueError(
                    f"id={identifier} has a non-diagnostic current report {original!r}; "
                    "supply --report-archive and/or --report-overrides"
                )
            replacement = impression_for_image(archive, image_list(record)[0])
            source = "archive_impression"
        if replacement is not None:
            record["report"] = replacement
            recovered.append({"id": record_id(record), "source": source,
                              "original_report": original, "replacement_report": replacement})
    return output, recovered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE_RECORDS)
    parser.add_argument("--report-archive", type=Path)
    parser.add_argument("--report-overrides", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    overrides = load_overrides(args.report_overrides)
    if args.start < 0 or (args.end is not None and args.end < args.start):
        raise ValueError("invalid selected range")
    if args.report_archive:
        with zipfile.ZipFile(args.report_archive) as archive:
            output, audit = recover_records(records, archive=archive, overrides=overrides,
                                            start=args.start, end=args.end)
    else:
        output, audit = recover_records(records, archive=None, overrides=overrides,
                                        start=args.start, end=args.end)
    if (args.output.exists() or args.audit_output.exists()) and not args.overwrite:
        raise FileExistsError("report recovery outputs already exist")
    dump_json(args.output, output, overwrite=args.overwrite)
    dump_json(args.audit_output, audit, overwrite=args.overwrite)
    print(f"wrote {len(output)} records, recovered {len(audit)} current reports")


if __name__ == "__main__":
    main()
