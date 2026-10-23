from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from data_generation.run_pipeline import (
    PipelinePaths,
    _manifest,
    _prepare_work_dir,
    _validate_final_annotations,
    _validate_inputs,
    _validate_path_separation,
    build_stages,
    parse_args,
)


class IntegratedPipelineTest(unittest.TestCase):
    def test_default_template_content_changes_invalidate_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / 'default_prompt.txt'
            template.write_text('Default task A')
            args = self._arguments(root)
            count, _ = _validate_inputs(args)
            paths = PipelinePaths.under(args.work_dir)
            with patch('data_generation.run_pipeline.DEFAULT_REASONING_PROMPT', template):
                manifest = _manifest(args, count)
                _prepare_work_dir(args, paths, manifest)
                stat = template.stat()
                template.write_text('Default task B')
                os.utime(template, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                args.resume = True
                with self.assertRaises(ValueError):
                    _prepare_work_dir(args, paths, _manifest(args, count))

    def test_custom_prompt_is_forwarded_and_content_changes_invalidate_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'prompt.json'
            config.write_text(json.dumps({'instruction': 'Task A', 'developer_instruction': 'Expert'}))
            args = self._arguments(root, '--reasoning-prompt-config', str(config))
            count, partial = _validate_inputs(args)
            paths = PipelinePaths.under(args.work_dir)
            stage = build_stages(args, paths, partial_selection=partial)[0]
            self.assertIn('--prompt-config', stage.command)
            self.assertIn(str(config.resolve()), stage.command)
            manifest = _manifest(args, count)
            _prepare_work_dir(args, paths, manifest)
            stat = config.stat()
            config.write_text(config.read_text().replace('Task A', 'Task B'))
            os.utime(config, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(config.stat().st_size, stat.st_size)
            args.resume = True
            with self.assertRaises(ValueError):
                _prepare_work_dir(args, paths, _manifest(args, count))

    def _arguments(self, root: Path, *extra: str):
        files = {
            "base.json": [
                {
                    "id": 1,
                    "image": ["files/p10/p100/s100/current.jpg"],
                    "report": "Report.",
                    "reasoning": "Reasoning.",
                },
                {
                    "id": 2,
                    "image": ["files/p10/p100/s101/current.jpg"],
                    "report": "Report two.",
                    "reasoning": "Reasoning two.",
                },
            ],
            "reports.json": [],
            "context.json": {"s100": {"Indication": ""}},
            "few_shot.json": [
                {"id": "example-1", "reasoning": "Example reasoning."}
            ],
            "anatomy.json": [
                {
                    "image": "files/p10/p100/s100/current.jpg",
                    "disease": [],
                }
            ],
        }
        for name, value in files.items():
            (root / name).write_text(json.dumps(value), encoding="utf-8")
        image_root = root / "images"
        image_root.mkdir()
        return parse_args(
            [
                "--base-records",
                str(root / "base.json"),
                "--reference-reports",
                str(root / "reports.json"),
                "--clinical-context",
                str(root / "context.json"),
                "--few-shot-examples",
                str(root / "few_shot.json"),
                "--anatomy-source",
                str(root / "anatomy.json"),
                "--image-root",
                str(image_root),
                "--model",
                "gpt-oss-test",
                "--work-dir",
                str(root / "work"),
                "--output",
                str(root / "final.json"),
                *extra,
            ]
        )

    def test_stage_order_covers_both_gpt_passes_and_all_transforms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._arguments(root)
            selected, partial = _validate_inputs(args)
            stages = build_stages(args, PipelinePaths.under(args.work_dir), partial_selection=partial)

        self.assertEqual(selected, 2)
        self.assertFalse(partial)
        self.assertEqual(
            [stage.name for stage in stages],
            [
                "generate_reasoning",
                "merge_reasoning",
                "prepare_anatomy_boxes",
                "attach_anatomy_boxes",
                "insert_bboxes",
                "build_mcot_dataset",
                "add_object_refs",
                "convert_to_qwenvl",
            ],
        )
        self.assertIn("data_generation.generate_reasoning", stages[0].command)
        self.assertIn("--few-shot-examples", stages[0].command)
        self.assertIn("data_generation.insert_bboxes", stages[4].command)
        self.assertIn("--validate-coordinates", stages[5].command)

    def test_review_inputs_are_forwarded_and_cannot_be_silently_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            notes = root / 'notes.json'
            notes.write_text('{"1": "Keep the NG tip visibility limitation."}')
            args = self._arguments(root, '--evidence-review', '--review-notes', str(notes))
            selected, partial = _validate_inputs(args)
            stage = build_stages(args, PipelinePaths.under(args.work_dir), partial_selection=partial)[0]
            self.assertIn('--evidence-review', stage.command)
            self.assertIn(str(notes), stage.command)
            self.assertIsNotNone(_manifest(args, selected)['review_notes'])
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._arguments(root, '--review-notes', str(root / 'notes.json'))
            with self.assertRaisesRegex(ValueError, 'require --evidence-review'):
                _validate_inputs(args)

    def test_partial_range_and_normalized_anatomy_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._arguments(
                root,
                "--start",
                "1",
                "--end",
                "2",
                "--anatomy-is-normalized",
            )
            selected, partial = _validate_inputs(args)
            stages = build_stages(args, PipelinePaths.under(args.work_dir), partial_selection=partial)

        self.assertEqual(selected, 1)
        self.assertTrue(partial)
        self.assertNotIn("prepare_anatomy_boxes", [stage.name for stage in stages])
        merge = next(stage for stage in stages if stage.name == "merge_reasoning")
        self.assertIn("--allow-missing", merge.command)

    def test_raw_anatomy_csv_passes_integrated_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._arguments(root)
            anatomy_csv = root / "anatomy.csv"
            anatomy_csv.write_text(
                "image,bbox_coordinates,bbox_labels,bbox_phrase_exists\n"
                'files/p10/p100/s100/current.jpg,"[[1, 2, 3, 4]]","[1]","[True]"\n',
                encoding="utf-8",
            )
            args.anatomy_source = anatomy_csv

            selected, partial = _validate_inputs(args)
            stages = build_stages(
                args,
                PipelinePaths.under(args.work_dir),
                partial_selection=partial,
            )

        self.assertEqual(selected, 2)
        self.assertFalse(partial)
        prepare = next(stage for stage in stages if stage.name == "prepare_anatomy_boxes")
        self.assertIn(str(anatomy_csv.resolve()), prepare.command)

    def test_final_annotation_image_placeholder_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "final.json"
            output.write_text(
                json.dumps(
                    [
                        {
                            "id": 1,
                            "image": ["current.jpg", "current.jpg_1_2_3_4"],
                            "conversations": [
                                {"from": "human", "value": "<image>"},
                                {"from": "gpt", "value": "<image>"},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(_validate_final_annotations(output), 1)

            broken = json.loads(output.read_text(encoding="utf-8"))
            broken[0]["image"].append("extra.jpg")
            output.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(ValueError):
                _validate_final_annotations(output)

    def test_resume_manifest_detects_in_place_input_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._arguments(root)
            selected, _ = _validate_inputs(args)
            paths = PipelinePaths.under(args.work_dir)
            manifest = _manifest(args, selected)
            _prepare_work_dir(args, paths, manifest)

            args.resume = True
            _prepare_work_dir(args, paths, manifest)

            base = json.loads((root / "base.json").read_text(encoding="utf-8"))
            base[0]["report"] = "Changed in place."
            (root / "base.json").write_text(json.dumps(base), encoding="utf-8")
            changed_manifest = _manifest(args, selected)
            with self.assertRaises(ValueError):
                _prepare_work_dir(args, paths, changed_manifest)

    def test_pipeline_cannot_overwrite_an_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._arguments(root)
            args.output = args.base_records
            selected, partial = _validate_inputs(args)
            self.assertEqual(selected, 2)
            paths = PipelinePaths.under(args.work_dir)
            stages = build_stages(args, paths, partial_selection=partial)
            with self.assertRaises(ValueError):
                _validate_path_separation(args, paths, stages)

    def test_truncated_current_report_is_rejected_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._arguments(root)
            records = json.loads(args.base_records.read_text(encoding="utf-8"))
            records[0]["report"] = "At the"
            args.base_records.write_text(json.dumps(records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-diagnostic"):
                _validate_inputs(args)

    def test_report_archive_adds_recovery_before_both_gpt_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self._arguments(root)
            archive_path = root / "reports.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("files/p10/p100/s100.txt", "IMPRESSION:\nNo pneumothorax.\n")
            args.report_archive = archive_path
            selected, partial = _validate_inputs(args)
            paths = PipelinePaths.under(args.work_dir)
            stages = build_stages(args, paths, partial_selection=partial)
            self.assertEqual(selected, 2)
            self.assertEqual(stages[0].name, "recover_current_reports")
            self.assertIn(str(paths.recovered_base), stages[1].command)
            self.assertIn(str(paths.recovered_base), stages[2].command)
            build = next(stage for stage in stages if stage.name == "build_mcot_dataset")
            self.assertIn("--restore-missing-conclusion", build.command)


if __name__ == "__main__":
    unittest.main()
