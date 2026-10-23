from __future__ import annotations

import io
import unittest
import zipfile

from data_generation.build_mcot_dataset import (
    build_mcot_dataset,
    preserve_narrative_with_boxes,
    restore_missing_conclusion,
)
from data_generation.recover_current_reports import looks_incomplete, recover_records


class ReportRecoveryTest(unittest.TestCase):
    def test_detects_truncated_and_acquisition_only_reports(self) -> None:
        for text in (
            "Send the", "At the", "See below",
            "Portable AP upright chest radiograph ___ at 12:22",
            "Portable supine chest film ___ at 16:50 is submitted.",
        ):
            self.assertTrue(looks_incomplete(text), text)
        self.assertFalse(looks_incomplete("No acute cardiopulmonary abnormality."))

    def test_archive_impression_replaces_truncated_current_not_prior(self) -> None:
        current = "files/p10/p100/s100/current.jpg"
        records = [{"id": 7, "image": [current, "files/p10/p100/s099/prior.jpg"],
                    "report": "At the"}]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "files/p10/p100/s100.txt",
                "FINDINGS:\nAt the\n\nIMPRESSION:\nNo pneumothorax. "
                "Left opacity has improved.\n",
            )
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as archive:
            output, audit = recover_records(records, archive=archive, overrides={})
        self.assertEqual(output[0]["report"], "No pneumothorax. Left opacity has improved.")
        self.assertEqual(output[0]["image"], records[0]["image"])
        self.assertEqual(audit[0]["source"], "archive_impression")
        self.assertEqual(records[0]["report"], "At the")

    def test_requires_source_or_explicit_override_for_truncated_report(self) -> None:
        records = [{"id": 7, "image": ["files/p10/p100/s100/current.jpg"],
                    "report": "See below"}]
        with self.assertRaisesRegex(ValueError, "non-diagnostic"):
            recover_records(records, archive=None, overrides={})
        output, audit = recover_records(
            records, archive=None,
            overrides={"7": "There is no pneumothorax or pleural effusion."},
        )
        self.assertEqual(audit[0]["source"], "override")
        self.assertIn("no pneumothorax", output[0]["report"])


class BboxConclusionPreservationTest(unittest.TestCase):
    def test_places_lower_and_mid_lung_boxes_before_their_own_observations(self) -> None:
        source = ("There is left lower‑lobe atelectasis. "
                  "Linear opacity is seen in the left mid‑lung field. "
                  "Bilateral pleural effusions are present.")
        predicted = source.replace("Linear opacity", "<|box_start|>1 2 3 4<|box_end|>Linear opacity")
        predicted = predicted.replace("Bilateral pleural", "<|box_start|>5 6 7 8<|box_end|>Bilateral pleural")
        result = preserve_narrative_with_boxes(source, predicted, [
            {"bbox_coordinates": [1, 2, 3, 4], "bbox_labels": "left lower lung zone"},
            {"bbox_coordinates": [5, 6, 7, 8], "bbox_labels": "left mid lung zone"},
        ])
        self.assertTrue(result.startswith("<|box_start|>1 2 3 4<|box_end|>\nThere is left lower"))
        self.assertIn("<|box_start|>5 6 7 8<|box_end|>\nLinear opacity", result)
        self.assertNotIn("<|box_end|>\nBilateral pleural", result)

    def test_preserves_deleted_clinical_sentence_and_repairs_model_typo(self) -> None:
        source = ("The patient has a valve replacement.\n\n"
                  "The mediastinum is normal.\n\nThe left lung is clear.")
        predicted = ("<|box_start|>1 2 3 4<|box_end|>The mediatorsium is normal.\n\n"
                     "The left lung is clear.")
        result = preserve_narrative_with_boxes(source, predicted)
        self.assertTrue(result.startswith("The patient has a valve replacement."))
        self.assertIn("<|box_end|>\nThe mediastinum is normal.", result)
        self.assertNotIn("mediatorsium", result)

    def test_preserves_original_spacing_and_nonstandard_summary(self) -> None:
        source = "There are mid‑ and lower lung opacities.\n\nIn synthesis, these may reflect edema."
        predicted = "<|box_start|>1 2 3 4<|box_end|>There are mid‑and lower lung opacities."
        result = preserve_narrative_with_boxes(source, predicted)
        self.assertIn("mid‑ and lower", result)
        self.assertTrue(result.endswith("In synthesis, these may reflect edema."))

    def test_rejects_box_in_model_added_observation(self) -> None:
        source = ("The lungs are clear. The heart is normal. The pleural spaces are clear. "
                  "There is no pneumothorax or effusion. The mediastinum is normal.")
        predicted = ("The lungs are clear. <|box_start|>1 2 3 4<|box_end|>Unexpected fracture. "
                     "The heart is normal. The pleural spaces are clear. "
                     "There is no pneumothorax or effusion. The mediastinum is normal.")
        with self.assertRaisesRegex(ValueError, "anchor"):
            preserve_narrative_with_boxes(source, predicted)

    def test_rejects_heavily_rewritten_narrative(self) -> None:
        with self.assertRaisesRegex(ValueError, "rewrote too much"):
            preserve_narrative_with_boxes("The left lung is clear.", "Multiple displaced rib fractures.")

    def test_rejects_coordinates_outside_supplied_anatomy(self) -> None:
        source = [{"id": 1, "image": ["files/p10/p100/s100/current.jpg"],
                   "report": "Lungs are clear.", "reasoning": "The lungs are clear.",
                   "disease": [{"bbox_coordinates": [10, 20, 100, 200]}]}]
        with self.assertRaisesRegex(ValueError, "absent from the supplied anatomy"):
            build_mcot_dataset(source, [[{"id": 1, "reasoning":
                "<|box_start|>1 2 3 4<|box_end|>The lungs are clear."}]],
                allow_overwrite=False, allow_missing=False,
                validate_coordinates=True, preserve_source_text=True)

    def test_restores_summary_omitted_by_bbox_pass(self) -> None:
        source = "The left lung is clear.\n\nIn summary, no acute disease."
        bbox = "<|box_start|>1 2 3 4<|box_end|> The left lung is clear."
        result = restore_missing_conclusion(source, bbox)
        self.assertTrue(result.endswith("In summary, no acute disease."))
        self.assertEqual(result.count("<|box_start|>"), 1)

    def test_does_not_duplicate_existing_summary(self) -> None:
        source = "The left lung is clear.\n\nIn summary, no acute disease."
        bbox = "The left lung is clear.\n\nIn summary, no acute disease."
        self.assertEqual(restore_missing_conclusion(source, bbox), bbox)

    def test_does_not_duplicate_summary_with_trailing_continuation(self) -> None:
        source = "The left lung is clear.\n\nIn summary, no acute disease."
        bbox = (
            "The left lung is clear.\n\nIn summary, no acute disease.\n\n"
            "<|box_start|>1 2 3 4<|box_end|> The pleural spaces are clear."
        )
        self.assertEqual(restore_missing_conclusion(source, bbox), bbox)

    def test_does_not_duplicate_box_prefixed_summary(self) -> None:
        source = "The left lung is clear.\n\nIn summary, no acute disease."
        bbox = (
            "The left lung is clear.\n"
            "<|box_start|>1 2 3 4<|box_end|> In summary, no acute disease."
        )
        self.assertEqual(restore_missing_conclusion(source, bbox), bbox)


if __name__ == "__main__":
    unittest.main()
