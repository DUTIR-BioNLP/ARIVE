from __future__ import annotations

import unittest
from pathlib import Path

from inference.infer_mcot_abn import (
    MCOT_INSTRUCTION_PROMPT,
    build_abn_test_keys,
    build_image_to_findings,
    data_construct,
    normalize_image_path,
    select_abn_test_samples,
)


class MimicAbnInferenceDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.current = "files/p10/p100/s200/current.jpg"
        self.prior = "files/p10/p100/s100/prior.jpg"

    def test_annotation_indices_normalize_anchor_and_nested_prior_paths(self) -> None:
        annotation = {
            "train": [
                {
                    "anchor_scan": {"image_path": ["p10/p100/s200/current.jpg"]},
                    "findings": "Current findings.",
                    "prior_study": {
                        "image_path": ["p10/p100/s100/prior.jpg"],
                        "findings": "Prior findings.",
                    },
                }
            ],
            "val": [],
            "test": [
                {
                    "anchor_scan": {"image_path": ["p10/p100/s200/current.jpg"]},
                    "findings": "Abnormal current findings.",
                }
            ],
        }

        lookup = build_image_to_findings(annotation)

        self.assertEqual(lookup[self.current], "Abnormal current findings.")
        self.assertEqual(lookup[self.prior], "Prior findings.")
        self.assertEqual(build_abn_test_keys(annotation), {self.current})
        self.assertEqual(
            normalize_image_path(f"/private/MIMIC-CXR/{self.current}"), self.current
        )

    def test_select_abn_test_samples_preserves_evaluation_order(self) -> None:
        reports = [
            {"question_id": 1, "image": ["files/p10/p100/s999/no.jpg"]},
            {"question_id": 2, "image": [self.current, self.prior]},
        ]

        selected = select_abn_test_samples(reports, {self.current})

        self.assertEqual([row["question_id"] for row in selected], [2])

    def test_prompt_orders_prior_before_current_and_requests_abnormal_only(self) -> None:
        message = data_construct(
            {"question_id": 2, "image": [self.current, self.prior]},
            {self.prior: "Small right pleural effusion."},
            {"s200": {"Indication": "Shortness of breath"}},
            Path("/images"),
        )
        content = message["content"]
        image_items = [item["image"] for item in content if item["type"] == "image"]
        text = "\n".join(item["text"] for item in content if item["type"] == "text")

        self.assertEqual(
            image_items,
            [f"/images/{self.prior}", f"/images/{self.current}"],
        )
        self.assertIn("Indication: Shortness of breath", text)
        self.assertIn("Prior abnormal findings:Small right pleural effusion.", text)
        self.assertIn("abnormal findings only", text)

    def test_missing_prior_findings_degrades_to_current_only(self) -> None:
        message = data_construct(
            {"question_id": 2, "image": [self.current, self.prior]},
            {},
            {},
            Path("/images"),
        )
        content = message["content"]
        image_items = [item["image"] for item in content if item["type"] == "image"]
        text = "\n".join(item["text"] for item in content if item["type"] == "text")

        self.assertEqual(image_items, [f"/images/{self.current}"])
        self.assertNotIn("Prior chest X-ray", text)
        self.assertIn(MCOT_INSTRUCTION_PROMPT, text)


if __name__ == "__main__":
    unittest.main()
