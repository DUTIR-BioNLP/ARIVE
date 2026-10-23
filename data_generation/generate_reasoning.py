"""Stage 1: generate final context/prior-aware CXR reasoning with GPT-OSS."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Sequence

from project_paths import (
    BASE_RECORDS,
    CLINICAL_CONTEXT,
    FEW_SHOT_EXAMPLES,
    REASONING_SHARD,
    REFERENCE_REPORTS,
)

from .common import (
    add_generation_arguments,
    clinical_context_text,
    image_list,
    load_json_object,
    load_records,
    reasoning_text,
    record_id,
    report_text,
    run_harmony_stage,
    study_id_from_path,
)

DEFAULT_REASONING_PROMPT = Path(__file__).parent / 'prompts' / 'merged_reasoning.txt'
REASONING_INSTRUCTION = 'You are an expert chest X-ray radiologist.'

REVIEW_INSTRUCTION = (
    "You audit radiology annotations for evidence fidelity. Repair unsupported imaging "
    "claims and overconfident conclusions while preserving the complete regional walkthrough. "
    "Return only the requested JSON edits."
)


def normalize_frontal_view_labels(narrative: str) -> str:
    """Scope paired acquisition labels to the dataset's frontal-only image.

    Only explicit paired view labels are normalized. Anatomical phrases such
    as 'lateral chest wall' and all clinical observations remain untouched.
    """
    return re.sub(
        r"\b(PA|AP|frontal|postero[-‑ ]anterior|antero[-‑ ]posterior)"
        r"\s+(?:and|&)\s+(?:a\s+)?(?:left\s+)?lateral"
        r"(?:\s+chest)?\s+(?:views?|projections?|radiographs?)\b",
        r"\1 view",
        narrative,
        flags=re.IGNORECASE,
    )


def build_report_index(records: list[dict[str, Any]]) -> dict[str, str]:
    reports: dict[str, str] = {}
    for record in records:
        images = image_list(record)
        if images:
            reports[study_id_from_path(images[0])] = report_text(record)
    return reports


def format_few_shot_examples(
    records: list[dict[str, Any]],
    example_ids: Sequence[str] | None = None,
) -> str:
    """Extract and format reasoning narratives from a curated example file."""
    selected = records
    if example_ids:
        records_by_id: dict[str, dict[str, Any]] = {}
        for record in records:
            key = str(record_id(record))
            if key in records_by_id:
                raise ValueError(f"duplicate few-shot example id: {key}")
            records_by_id[key] = record

        missing = [key for key in example_ids if str(key) not in records_by_id]
        if missing:
            raise KeyError(f"few-shot example ids not found: {', '.join(map(str, missing))}")
        selected = [records_by_id[str(key)] for key in example_ids]

    if not selected:
        raise ValueError("few-shot example file contains no selected records")

    examples: list[str] = []
    for index, record in enumerate(selected, start=1):
        try:
            narrative = reasoning_text(record)
        except (KeyError, TypeError) as exc:
            identifier = record.get("id", record.get("question_id", index))
            raise ValueError(
                f"few-shot example id={identifier} has no readable reasoning narrative"
            ) from exc
        # Paired examples show which observations the narrative is allowed to
        # expand. Narrative-only legacy examples remain supported.
        source_report = record.get("report", record.get("findings"))
        if isinstance(source_report, str) and source_report.strip():
            evidence = f"Current report: {source_report.strip()}"
            example_context = record.get("clinical_context")
            if isinstance(example_context, str) and example_context.strip():
                evidence += f"\nClinical context: {example_context.strip()}"
            if "prior_report" in record:
                evidence += f"\nPrior report: {record['prior_report'] or 'None'}"
            examples.append(
                f"### Example {index}\n{evidence}\n"
                f"Reasoning: {narrative}"
            )
        else:
            examples.append(f"### Example {index}\n{narrative}")
    return "\n\n".join(examples)


def build_prompt(
    record: dict[str, Any],
    clinical_context: dict[str, Any],
    prior_reports: dict[str, str],
    few_shot_examples: str,
    prompt_config: dict[str, str] | None = None,
) -> str:
    images = image_list(record)
    if not images:
        raise ValueError("reasoning record has no image")
    current_image = images[0]
    prior_image = images[1] if len(images) > 1 else current_image
    study_id = study_id_from_path(current_image)
    context = clinical_context.get(study_id)
    if not isinstance(context, dict):
        raise KeyError(f"clinical context is missing study {study_id}")

    if not isinstance(few_shot_examples, str) or not few_shot_examples.strip():
        raise ValueError("few-shot examples must be a non-empty string")
    current_report = report_text(record)
    context_block = clinical_context_text(context)
    prior_report = "None"
    if current_image != prior_image:
        prior_id = study_id_from_path(prior_image)
        if prior_id not in prior_reports:
            raise KeyError(f"prior report is missing study {prior_id}")
        prior_report = prior_reports[prior_id]

    if prompt_config is not None:
        validate_prompt_config(prompt_config)
        return f"""<few_shot_examples>
{few_shot_examples}
</few_shot_examples>

<input>
Current CXR report:
{current_report}

Clinical context:
{context_block}

Prior CXR report:
{prior_report}
</input>

{prompt_config['instruction']}"""

    return DEFAULT_REASONING_PROMPT.read_text(encoding='utf-8').strip().format(
        FEW_SHOT_EXAMPLES=few_shot_examples,
        CURRENT_CXR_REPORT=current_report,
        CLINICAL_CONTEXT=context_block,
        PRIOR_CXR_REPORT_OR_NONE=prior_report,
    )


def validate_prompt_config(config: dict[str, Any]) -> None:
    for key in ('instruction', 'developer_instruction'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f'prompt config needs a non-empty {key}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_RECORDS,
        help="Records containing current/prior images and a current report.",
    )
    parser.add_argument("--clinical-context", type=Path, default=CLINICAL_CONTEXT)
    parser.add_argument('--prompt-config', type=Path,
                        help='JSON containing instruction and developer_instruction; preserves patient inputs and full examples.')
    parser.add_argument("--draft-reasoning", type=Path,
                        help="Revise a saved draft shard instead of generating new drafts.")
    parser.add_argument("--review-notes", type=Path,
                        help="Optional per-ID evidence-audit notes for sentence revision.")
    parser.add_argument("--skip-evidence-review", action="store_true",
                        help="Generate drafts only, without the evidence-fidelity revision.")
    parser.add_argument("--evidence-review", dest="skip_evidence_review", action="store_false",
                        help="Also revise evidence fidelity; this adds a generation pass.")
    parser.set_defaults(skip_evidence_review=True)
    parser.add_argument(
        "--reference-reports",
        type=Path,
        default=REFERENCE_REPORTS,
        help="Full report corpus used to resolve prior studies.",
    )
    parser.add_argument(
        "--few-shot-examples",
        type=Path,
        default=FEW_SHOT_EXAMPLES,
        help="Curated JSON/JSONL records containing example reasoning narratives.",
    )
    parser.add_argument(
        "--few-shot-example-id",
        dest="few_shot_example_ids",
        action="append",
        default=[],
        help="Optional example id to select, in display order; repeat for multiple ids.",
    )
    add_generation_arguments(parser, default_output=REASONING_SHARD)
    parser.set_defaults(max_model_len=20480, max_tokens=10240)
    return parser.parse_args()


def build_review_prompt(source_prompt: str, draft: str, review_notes: str = '') -> str:
    from .revise_evidence import build_revision_prompt
    evidence = source_prompt.split("<input>", 1)[1].split("</input>", 1)[0]
    return build_revision_prompt(evidence, draft, review_notes)


def main() -> None:
    from .revise_evidence import apply_sentence_edits
    args = parse_args()
    prompt_config = load_json_object(args.prompt_config) if args.prompt_config else None
    if prompt_config is not None:
        validate_prompt_config(prompt_config)
    records = load_records(args.input)
    context = load_json_object(args.clinical_context)
    prior_reports = build_report_index(load_records(args.reference_reports))
    few_shot_examples = format_few_shot_examples(
        load_records(args.few_shot_examples),
        args.few_shot_example_ids,
    )
    if args.draft_reasoning and args.skip_evidence_review:
        raise ValueError("saved drafts require evidence revision")
    if args.review_notes and args.skip_evidence_review:
        raise ValueError("review notes require --evidence-review")
    review_notes = load_json_object(args.review_notes) if args.review_notes else {}
    saved_drafts = None
    if args.draft_reasoning:
        draft_records = load_records(args.draft_reasoning)
        saved_drafts = {
            str(record_id(r)): (r["response"] if isinstance(r.get("response"), str)
                               and r["response"].strip() else reasoning_text(r))
            for r in draft_records
        }
        if len(saved_drafts) != len(draft_records):
            raise ValueError("duplicate ids in saved drafts")
    make_prompt = lambda record: build_prompt(record, context, prior_reports, few_shot_examples, prompt_config)
    run_harmony_stage(
        records=records,
        prompt_builder=make_prompt,
        args=args,
        output_transform=normalize_frontal_view_labels,
        developer_instruction=(prompt_config['developer_instruction'] if prompt_config else REASONING_INSTRUCTION),
        raw_output_path=args.output.parent / "drafts" / args.output.name,
        saved_drafts=saved_drafts,
        review_prompt_builder=(None if args.skip_evidence_review else
                               lambda record, draft: build_review_prompt(
                                   make_prompt(record), draft, review_notes.get(str(record_id(record)), ''))),
        review_developer_instruction=REVIEW_INSTRUCTION,
        review_output_transform=(None if args.skip_evidence_review else
                                 lambda record, draft, response: apply_sentence_edits(
                                     {"reasoning": draft}, response)),
        review_raw_output_path=(None if args.skip_evidence_review else
                                args.output.parent / "review" / args.output.name),
    )


if __name__ == "__main__":
    main()
