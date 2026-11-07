from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from data_generation.add_object_refs import add_object_refs, build_bbx_by_id
from data_generation.attach_anatomy_boxes import attach_boxes
from data_generation.box_utils import (
    coordinates,
    normalize_reasoning_boxes,
    validate_normalized_coordinates,
)
from data_generation.convert_to_qwenvl import convert_record
from data_generation.generate_reasoning import (
    build_prompt as build_reasoning_prompt,
    build_review_prompt,
    format_few_shot_examples,
    normalize_frontal_view_labels,
)
from data_generation.merge_reasoning import merge_reasoning
from data_generation.insert_bboxes import insert_selected_anchors, insert_selected_ids, sentence_spans
from data_generation.prepare_anatomy_boxes import normalize_box, prepare_record
from data_generation.revise_evidence import apply_sentence_edits


CURRENT = "files/p10/p100/s100/current.jpg"
PRIOR = "files/p10/p100/s099/prior.jpg"


class BoxUtilitiesTest(unittest.TestCase):
    def test_repairs_deduplicates_and_validates_boxes(self) -> None:
        raw = (
            "<|box_end|>first <|box_start|>10 20 100 200<|box_end> "
            "duplicate <|box_start|>10 20 100 200<|box_end|>"
        )
        normalized = normalize_reasoning_boxes(raw)
        self.assertEqual(coordinates(normalized), [(10, 20, 100, 200)])
        self.assertNotIn("<|box_end>", normalized)
        validate_normalized_coordinates(normalized)

    def test_rejects_invalid_coordinate_geometry(self) -> None:
        with self.assertRaises(ValueError):
            validate_normalized_coordinates("<|box_start|>20 20 10 30<|box_end|>")


class DatasetJoinTest(unittest.TestCase):
    def test_reasoning_shards_are_joined_by_id_not_position(self) -> None:
        base = [
            {"id": 1, "findings": "one", "image": [CURRENT]},
            {"id": 2, "findings": "two", "image": [CURRENT]},
        ]
        shards = [[{"id": 2, "reasoning": "r2"}, {"id": 1, "reasoning": "r1"}]]
        merged = merge_reasoning(
            base,
            shards,
            allow_overwrite=False,
            allow_missing=False,
        )
        self.assertEqual([item["id"] for item in merged], [1, 2])
        self.assertEqual([item["reasoning"] for item in merged], ["r1", "r2"])

    def test_conflicting_correction_requires_explicit_permission(self) -> None:
        base = [{"id": 1, "report": "one", "image": [CURRENT]}]
        shards = [[{"id": 1, "reasoning": "old"}], [{"id": 1, "reasoning": "new"}]]
        with self.assertRaises(ValueError):
            merge_reasoning(base, shards, allow_overwrite=False, allow_missing=False)
        merged = merge_reasoning(base, shards, allow_overwrite=True, allow_missing=False)
        self.assertEqual(merged[0]["reasoning"], "new")

    def test_object_references_match_across_string_and_integer_ids(self) -> None:
        bbox_source = [
            {
                "id": "7",
                "disease": [
                    {"bbox_labels": "right lung", "bbox_coordinates": [10, 20, 100, 200]}
                ],
            }
        ]
        mcot = [
            {
                "id": 7,
                "reasoning": "Inspect <|box_start|>10 20 100 200<|box_end|> now.",
            }
        ]
        updated, stats = add_object_refs(
            copy.deepcopy(mcot),
            build_bbx_by_id(bbox_source),
            join_labels_with="; ",
        )
        self.assertIn(
            "<|object_ref_start|>right lung<|object_ref_end|>",
            updated[0]["reasoning"],
        )
        self.assertEqual(stats["matched_boxes"], 1)


class ReasoningPromptTest(unittest.TestCase):
    def test_custom_instructions_retain_complete_examples_and_patient_inputs(self):
        examples = 'Example paragraph one.\n\nExample paragraph two.'
        config = {'instruction': 'Generate the merged narrative.',
                  'developer_instruction': 'You are a radiologist editor.'}
        prompt = build_reasoning_prompt(
            {'id': 1, 'image': [CURRENT, PRIOR], 'report': 'Current finding.'},
            {'s100': {'Indication': 'Dyspnea', 'History': 'History text.',
                      'Technique': 'Portable AP', 'Comparison': 'Yesterday'}},
            {'s099': 'Prior finding.'}, examples, config,
        )
        for value in [examples, 'Current finding.', 'Prior finding.', 'Dyspnea',
                      'History text.', 'Portable AP', 'Yesterday']:
            self.assertIn(value, prompt)
        self.assertTrue(prompt.endswith(config['instruction']))
        audit = build_review_prompt(prompt, 'A generated draft.')
        self.assertIn('Current finding.', audit)
        self.assertIn('Prior finding.', audit)
        self.assertNotIn(examples, audit)
        with self.assertRaisesRegex(ValueError, 'developer_instruction'):
            build_reasoning_prompt(
                {'id': 1, 'image': [CURRENT], 'report': 'Current finding.'},
                {'s100': {}}, {}, examples, {'instruction': 'A task.'},
            )

    def test_frontal_label_normalization_preserves_clinical_lateral_anatomy(self) -> None:
        text = ("I review PA and lateral views. Possible left pleural effusion. "
                "A lesion lies along the lateral chest wall. The tube is 6 cm above the carina.")
        normalized = normalize_frontal_view_labels(text)
        self.assertEqual(normalized, text.replace("PA and lateral views", "PA view"))
        self.assertEqual(normalize_frontal_view_labels(normalized), normalized)

    def test_first_gpt_pass_combines_few_shot_generation_context_and_prior(self) -> None:
        prompt = build_reasoning_prompt(
            {
                "id": 1,
                "image": [CURRENT, PRIOR],
                "report": "Current findings.",
            },
            {
                "s100": {
                    "Indication": "Dyspnea",
                    "History": "Example history",
                    "Technique": "Portable AP",
                    "Comparison": "Prior study",
                }
            },
            {"s099": "Prior findings."},
            "### Example 1\nExample reasoning.",
        )
        self.assertIn("<few_shot_examples>", prompt)
        self.assertIn("Example reasoning.", prompt)
        self.assertIn("Current findings.", prompt)
        self.assertIn("Prior findings.", prompt)
        self.assertIn("History: Example history", prompt)
        self.assertIn("Technique: Portable AP", prompt)
        self.assertIn("Comparison: Prior study", prompt)
        self.assertNotIn("Base reasoning narrative", prompt)

    def test_first_gpt_pass_marks_absent_prior_as_none(self) -> None:
        prompt = build_reasoning_prompt(
            {"id": 1, "image": [CURRENT], "report": "Current findings."},
            {"s100": {}},
            {},
            "### Example 1\nExample reasoning.",
        )
        self.assertIn("Prior CXR report:\nNone", prompt)
        self.assertNotIn('History:', prompt)
        self.assertNotIn('Technique:', prompt)
        self.assertNotIn('Indication:', prompt)

    def test_revision_uses_patient_evidence_and_draft_without_few_shot_facts(self) -> None:
        source = build_reasoning_prompt(
            {"id": 1, "image": [CURRENT, PRIOR], "report": "Current findings."},
            {"s100": {"Indication": "Dyspnea"}},
            {"s099": "Prior findings."}, "Other patient's findings.",
        )
        revised = build_review_prompt(source, "Generated draft.")
        self.assertIn("Current findings.", revised)
        self.assertIn("Prior findings.", revised)
        self.assertIn("Indication: Dyspnea", revised)
        self.assertIn("Generated draft.", revised)
        self.assertNotIn("Other patient's findings.", revised)

    def test_few_shot_examples_are_selected_and_formatted_in_requested_order(self) -> None:
        examples = format_few_shot_examples(
            [
                {"id": "a", "reasoning": "First reasoning."},
                {"id": "b", "reasoning": "Second reasoning."},
            ],
            ["b", "a"],
        )
        self.assertEqual(
            examples,
            "### Example 1\nSecond reasoning.\n\n### Example 2\nFirst reasoning.",
        )

    def test_paired_example_keeps_report_evidence_with_its_narrative(self) -> None:
        examples = format_few_shot_examples([
            {"id": "paired", "report": "Possible small effusion.",
             "clinical_context": "Indication: Dyspnea.", "prior_report": None,
             "reasoning": "I note possible pleural fluid, retaining the uncertainty."},
            {"id": "legacy", "reasoning": "Legacy narrative."},
        ])
        self.assertIn("Current report: Possible small effusion.", examples)
        self.assertIn("Clinical context: Indication: Dyspnea.\nPrior report: None\nReasoning: I note", examples)
        self.assertTrue(examples.endswith("### Example 2\nLegacy narrative."))


class AnatomyPreparationTest(unittest.TestCase):
    def test_normalization_uses_width_and_height_with_either_legacy_flag(self) -> None:
        box = [10, 20, 60, 100]
        self.assertEqual(
            normalize_box(box, width=100, height=200, legacy_swapped_dimensions=False),
            [100, 100, 600, 500],
        )
        self.assertEqual(
            normalize_box(box, width=100, height=200, legacy_swapped_dimensions=True),
            [100, 100, 600, 500],
        )

    def test_prepare_and_attach_anatomy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / CURRENT
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (100, 200)).save(image_path)
            anatomy = prepare_record(
                {
                    "image": CURRENT,
                    "bbox_coordinates": [[10, 20, 60, 100]],
                    "bbox_labels": [1],
                    "bbox_phrase_exists": [True],
                },
                image_root=root,
                keep_all=False,
                legacy_swapped_dimensions=False,
            )

        self.assertEqual(anatomy["disease"][0]["bbox_labels"], "right lung")
        matched, missing = attach_boxes(
            [{"id": 1, "image": [CURRENT], "reasoning": "r", "report": "f"}],
            [anatomy],
        )
        self.assertEqual(len(matched), 1)
        self.assertEqual(missing, [])
        self.assertEqual(matched[0]["disease"], anatomy["disease"])


class BboxSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {
            "reasoning": "I assess the right lung. Minimal right basal atelectasis is present.\n\nThe left lung is clear.",
            "disease": [
                {"bbox_labels": "right lower lung zone", "bbox_coordinates": [10, 20, 100, 200]},
                {"bbox_labels": "left lung", "bbox_coordinates": [300, 20, 500, 600]},
            ],
        }

    def test_local_observation_anchors_preserve_every_source_character(self) -> None:
        selections = [
            {"region": "right lower lung zone", "before": "Minimal right basal atelectasis is present."},
            {"region": "left lung", "before": "The left lung is clear."},
        ]
        result = insert_selected_anchors(self.record, json.dumps(selections))
        self.assertEqual(result.replace("<|box_start|>10 20 100 200<|box_end|>\n", "")
                         .replace("<|box_start|>300 20 500 600<|box_end|>\n", ""), self.record['reasoning'])
        self.assertIn("<|box_end|>\nMinimal right basal atelectasis", result)

    def test_numbered_selection_handles_decimal_and_quoted_sentences(self) -> None:
        self.record['reasoning'] = 'The tip is 6.8 cm above the carina. The report states “clear lungs.”\n\nThe left lung is clear.'
        sentences = [self.record['reasoning'][start:end] for start, end in sentence_spans(self.record['reasoning'])]
        self.assertEqual(sentences, ['The tip is 6.8 cm above the carina.',
                                    'The report states “clear lungs.”', 'The left lung is clear.'])
        result = insert_selected_ids(self.record, '[{"region_id": 1, "sentence_id": 2}]')
        self.assertEqual(result.replace('<|box_start|>300 20 500 600<|box_end|>\n', ''), self.record['reasoning'])

    def test_numbered_selection_rejects_invalid_or_duplicate_ids(self) -> None:
        for selections in [
            [{'region_id': 4, 'sentence_id': 0}], [{'region_id': 1, 'sentence_id': 20}],
            [{'region_id': True, 'sentence_id': 0}],
            [{'region_id': 1, 'sentence_id': 0}, {'region_id': 1, 'sentence_id': 2}],
            [{'region_id': 0, 'sentence_id': 1}, {'region_id': 1, 'sentence_id': 1}],
        ]:
            with self.subTest(selections=selections), self.assertRaises(ValueError):
                insert_selected_ids(self.record, json.dumps(selections))

    def test_rejects_hallucinated_region_or_rewritten_anchor(self) -> None:
        for selection in [
            {"region": "carina", "before": "The left lung is clear."},
            {"region": "left lung", "before": "The left lung is normal."},
            {"region": "left lung", "before": "left lung is clear."},
        ]:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                insert_selected_anchors(self.record, json.dumps([selection]))

    def test_sentence_after_quoted_observation_is_a_valid_anchor(self) -> None:
        self.record['reasoning'] = 'The report states “clear lungs.” The left lung is clear.'
        result = insert_selected_anchors(self.record, json.dumps([
            {"region": "left lung", "before": "The left lung is clear."},
        ]))
        self.assertIn('“clear lungs.” <|box_start|>300 20 500 600<|box_end|>', result)

    def test_minor_tail_typo_uses_unique_literal_prefix_without_editing_source(self) -> None:
        self.record['reasoning'] = 'In the left lung field the report notes minimal basal atelectasis.'
        result = insert_selected_anchors(self.record, json.dumps([
            {'region': 'left lung', 'before': 'In the left lung field the report note minimal basal atelectasis'},
        ]))
        self.assertTrue(result.endswith(self.record['reasoning']))

    def test_rejects_duplicate_boxes_and_ambiguous_quotes(self) -> None:
        selection = {"region": "left lung", "before": "The left lung is clear."}
        with self.assertRaises(ValueError):
            insert_selected_anchors(self.record, json.dumps([selection, selection]))
        self.record['reasoning'] += '\n\nThe left lung is clear.'
        with self.assertRaises(ValueError):
            insert_selected_anchors(self.record, json.dumps([selection]))

    def test_two_regions_for_one_sentence_create_one_crop_observation(self) -> None:
        self.record['reasoning'] = 'The cardiac and mediastinal contours are stable.'
        self.record['disease'] = [
            {'bbox_labels': 'mediastinum', 'bbox_coordinates': [10, 20, 100, 200]},
            {'bbox_labels': 'cardiac silhouette', 'bbox_coordinates': [300, 20, 500, 600]},
        ]
        result = insert_selected_anchors(self.record, json.dumps([
            {'region': 'mediastinum', 'before': self.record['reasoning']},
            {'region': 'cardiac silhouette', 'before': self.record['reasoning']},
        ]))
        self.assertEqual(result.count('<|box_start|>'), 1)


class QwenConversionTest(unittest.TestCase):
    def _record(self, images: list[str]) -> dict:
        return {
            "id": 11,
            "image": images,
            "report": "No focal airspace opacity.",
            "reasoning": (
                "I inspect the lungs. "
                "<|object_ref_start|>right lung<|object_ref_end|>"
                "<|box_start|>10 20 100 200<|box_end|> The region is clear."
            ),
        }

    def test_current_only_annotation_and_virtual_crop(self) -> None:
        converted = convert_record(
            self._record([CURRENT]),
            {"s100": {"Indication": "Dyspnea"}},
            {},
            require_object_refs=True,
        )
        self.assertEqual(
            converted["image"],
            [CURRENT, f"{CURRENT}_10_20_100_200"],
        )
        assistant = converted["conversations"][1]["value"]
        self.assertIn("<|box_end|>\n<image>\n The region is clear.", assistant)
        self.assertEqual(
            sum(turn["value"].count("<image>") for turn in converted["conversations"]),
            len(converted["image"]),
        )

    def test_prior_image_precedes_current_image(self) -> None:
        converted = convert_record(
            self._record([CURRENT, PRIOR]),
            {"s100": {"Indication": "Follow-up"}},
            {"s099": "Prior findings."},
            require_object_refs=True,
        )
        self.assertEqual(converted["image"][:2], [PRIOR, CURRENT])
        self.assertEqual(len(converted["image"]), 3)

    def test_training_input_uses_only_indication_with_or_without_prior(self) -> None:
        context = {
            "Indication": "Dyspnea",
            "History": "Postoperative fluid resuscitation.",
            "Technique": "Portable AP",
            "Comparison": "Two hours earlier.",
        }
        for images in ([CURRENT], [CURRENT, PRIOR]):
            with self.subTest(images=images):
                converted = convert_record(
                    self._record(images), {"s100": context},
                    {"s099": "Prior findings."}, require_object_refs=True,
                )
                user = converted["conversations"][0]["value"]
                self.assertIn("Indication: Dyspnea\n", user)
                for field in ("History", "Technique", "Comparison"):
                    self.assertNotIn(f"{field}:", user)
                    self.assertNotIn(context[field], user)
                if len(images) > 1:
                    self.assertIn("Prior findings:Prior findings.", user)
                else:
                    self.assertNotIn("Prior findings:", user)
                self.assertIn(
                    "<|object_ref_start|>right lung",
                    converted["conversations"][1]["value"],
                )

    def test_missing_indication_does_not_fall_back_to_other_context(self) -> None:
        for indication_fields in (
            {}, {"Indication": None}, {"Indication": ""},
            {"Indication": "  "}, {"Indication": float("nan")},
        ):
            for images in ([CURRENT], [CURRENT, PRIOR]):
                with self.subTest(indication=indication_fields, images=images):
                    context = {
                        "History": "Postoperative fluid resuscitation.",
                        "Technique": "Portable AP",
                        "Comparison": "Two hours earlier.",
                        **indication_fields,
                    }
                    converted = convert_record(
                        self._record(images), {"s100": context},
                        {"s099": "Prior findings."}, require_object_refs=True,
                    )
                    user = converted["conversations"][0]["value"]
                    for field in ("Indication", "History", "Technique", "Comparison"):
                        self.assertNotIn(f"{field}:", user)
                    for field in ("History", "Technique", "Comparison"):
                        self.assertNotIn(context[field], user)


class EvidenceRevisionTest(unittest.TestCase):
    def test_edit_preserves_all_unselected_sentences_and_paragraphs(self) -> None:
        source = 'I assess the tube. It is safe and without kinking.\n\nThe lungs are clear.'
        result = apply_sentence_edits({'reasoning': source}, json.dumps([
            {'sentence_id': 1, 'replacement': 'The tip terminates in the mid SVC.'},
        ]))
        self.assertEqual(result, source.replace('It is safe and without kinking.', 'The tip terminates in the mid SVC.'))

    def test_review_cannot_delete_a_sentence_or_introduce_crop_tags(self) -> None:
        for replacement in ['', '<|box_start|>1 2 3 4<|box_end|>']:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                apply_sentence_edits({'reasoning': 'The lungs are clear.'}, json.dumps([
                    {'sentence_id': 0, 'replacement': replacement},
                ]))


if __name__ == "__main__":
    unittest.main()
