"""Revise unsupported sentences without regenerating a complete narrative."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import (
    add_generation_arguments, dump_json, image_list, load_json_object,
    load_records, record_id, reasoning_text, run_harmony_stage,
)
from .generate_reasoning import build_prompt, build_report_index
from .insert_bboxes import sentence_spans

REVIEW_INSTRUCTION = (
    "You audit radiology annotations for evidence fidelity. "
    "Repair unsupported imaging claims and overconfident conclusions while preserving "
    "the complete regional walkthrough. Return only the requested JSON edits."
)


def build_revision_prompt(evidence: str, draft: str, review_notes: str = '') -> str:
    sentences = '\n'.join(f'{i}: {draft[start:end]}'
                          for i, (start, end) in enumerate(sentence_spans(draft)))
    return f"""Audit each numbered draft sentence against the supplied current report, clinical context and prior. Return a JSON array of {{"sentence_id": integer, "replacement": "complete corrected sentence(s)", "reason": "unsupported claim being corrected"}}. Return [] only if no correction is needed.

Correct invented appearances, devices/quality inferred from silence, unsupported positioning verdicts, prior-only details carried forward, and conclusions that exceed the evidence. Reported findings can have conditional clinical explanations, but normal contours do not exclude underlying diseases. Preserve every source qualifier, measured position, limitation and interval finding. Use the frontal examination only. Keep substantial regional reasoning; replace unsupported detail with supported interpretation, rather than reducing the draft to a short report paraphrase. Avoid commentary about inputs or instructions.

Examples:
Current: PICC ends in mid SVC. Draft: The PICC is correctly placed without kinking. Replacement: The PICC terminates in the mid SVC, providing the reported anatomical endpoint for the device assessment.
Current: Cardiomediastinal contours stable. Draft: The heart is normal with no pericardial effusion. Replacement: The cardiomediastinal contours remain stable; this preserves their reported interval appearance without establishing normal cardiac size or function.
Current: Lungs clear. Draft: The normal hila exclude embolic disease. Replacement: The lungs are clear, providing no demonstrated parenchymal abnormality while not excluding disease that may lack a radiographic correlate.

<evidence>
{evidence}
</evidence>
<reviewer_notes>
{review_notes or 'Review all sentences for unsupported additions.'}
</reviewer_notes>
<sentences>
{sentences}
</sentences>

The reviewer notes identify concrete defects: correct every affected sentence, including repeated conclusions. Do not replace one unsupported claim with another. Return only the JSON edits, retaining the remainder of the narrative unchanged."""


def apply_sentence_edits(record: dict[str, Any], response: str) -> str:
    if response.startswith('```json\n') and response.endswith('\n```'):
        response = response[8:-4]
    edits = json.loads(response)
    if not isinstance(edits, list):
        raise ValueError('evidence edits must be a JSON array')
    source = reasoning_text(record)
    spans = sentence_spans(source)
    replacements = {}
    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError('each evidence edit must be an object')
        index, replacement = edit.get('sentence_id'), edit.get('replacement')
        if type(index) is not int or not 0 <= index < len(spans):
            raise ValueError('invalid evidence-edit sentence ID')
        if index in replacements:
            raise ValueError('conflicting edits for one sentence')
        if not isinstance(replacement, str) or not replacement.strip():
            raise ValueError('a correction must retain a supported observation or interpretation')
        if '<|' in replacement or '<image>' in replacement:
            raise ValueError('evidence revision cannot introduce grounding tags')
        replacements[index] = replacement.strip()
    result = source
    for index, replacement in sorted(replacements.items(), reverse=True):
        start, end = spans[index]
        result = result[:start] + replacement + result[end:]
    if len(result.split()) < min(300, int(0.75 * len(source.split()))):
        raise ValueError('evidence revision compressed the full narrative')
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--base-records', type=Path, required=True)
    parser.add_argument('--clinical-context', type=Path, required=True)
    parser.add_argument('--reference-reports', type=Path, required=True)
    parser.add_argument('--review-notes', type=Path,
                        help='Optional JSON object of case IDs to evidence-audit notes, not reference reasoning.')
    add_generation_arguments(parser)
    parser.set_defaults(max_model_len=20480, max_tokens=10240)
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_records(args.input)
    base = {str(record_id(r)): r for r in load_records(args.base_records)}
    context = load_json_object(args.clinical_context)
    priors = build_report_index(load_records(args.reference_reports))
    review_notes = load_json_object(args.review_notes) if args.review_notes else {}

    def prompt(record):
        source_prompt = build_prompt(base[str(record_id(record))], context, priors, 'Style omitted for audit.')
        evidence = source_prompt.split('<input>', 1)[1].split('</input>', 1)[0]
        return build_revision_prompt(evidence, reasoning_text(record),
                                     review_notes.get(str(record_id(record)), ''))

    run_harmony_stage(
        records=records, prompt_builder=prompt, args=args,
        developer_instruction=REVIEW_INSTRUCTION,
        record_output_transform=apply_sentence_edits,
        raw_output_path=args.output.parent / 'raw' / args.output.name,
    )


if __name__ == '__main__':
    main()
