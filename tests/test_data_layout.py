from __future__ import annotations

import unittest

from data_generation.common import load_records
from data_generation.prepare_anatomy_boxes import load_source, prepare_record
from data_generation.run_pipeline import (
    PipelinePaths,
    _validate_final_annotations,
    _validate_inputs,
    build_stages,
    parse_args,
)
from project_paths import (
    ANATOMY_SOURCE,
    BASE_RECORDS,
    CLINICAL_CONTEXT,
    DATA_ROOT,
    FEW_SHOT_EXAMPLES,
    IMAGE_ROOT,
    MCOT_ANNOTATION,
    PIPELINE_WORK_DIR,
    REFERENCE_REPORTS,
)


class DataLayoutTest(unittest.TestCase):
    def test_integrated_pipeline_defaults_to_data_tree(self) -> None:
        args = parse_args(["--model", "gpt-oss-test"])

        self.assertEqual(args.base_records, BASE_RECORDS)
        self.assertEqual(args.reference_reports, REFERENCE_REPORTS)
        self.assertEqual(args.clinical_context, CLINICAL_CONTEXT)
        self.assertEqual(args.few_shot_examples, FEW_SHOT_EXAMPLES)
        self.assertEqual(args.anatomy_source, ANATOMY_SOURCE)
        self.assertEqual(args.image_root, IMAGE_ROOT)
        self.assertEqual(args.work_dir, PIPELINE_WORK_DIR)
        self.assertEqual(args.output, MCOT_ANNOTATION)

    def test_tracked_synthetic_example_is_self_consistent(self) -> None:
        sample = DATA_ROOT / "example_synthetic"
        inputs = sample / "inputs"
        args = parse_args(
            [
                "--base-records",
                str(inputs / "base_records.json"),
                "--reference-reports",
                str(inputs / "reference_reports.json"),
                "--clinical-context",
                str(inputs / "clinical_context.json"),
                "--few-shot-examples",
                str(inputs / "few_shot_examples.json"),
                "--anatomy-source",
                str(inputs / "anatomy_boxes.json"),
                "--image-root",
                str(inputs / "images"),
                "--model",
                "gpt-oss-test",
                "--work-dir",
                str(sample / "work" / "test"),
                "--output",
                str(sample / "output" / "test.json"),
                "--dry-run",
            ]
        )

        selected, partial = _validate_inputs(args)
        stages = build_stages(
            args,
            PipelinePaths.under(args.work_dir),
            partial_selection=partial,
        )
        self.assertEqual(selected, 1)
        self.assertFalse(partial)
        self.assertEqual(len(stages), 8)

        for record in load_records(inputs / "base_records.json"):
            for relative_image in record["image"]:
                self.assertTrue((inputs / "images" / relative_image).is_file())

        anatomy = load_source(inputs / "anatomy_boxes.json")
        normalized = prepare_record(
            anatomy[0],
            image_root=inputs / "images",
            keep_all=False,
            legacy_swapped_dimensions=False,
        )
        self.assertEqual(
            normalized["disease"][0]["bbox_coordinates"],
            [500, 500, 1000, 1000],
        )
        self.assertEqual(
            _validate_final_annotations(sample / "expected" / "mcot_annotation.json"),
            1,
        )


if __name__ == "__main__":
    unittest.main()
