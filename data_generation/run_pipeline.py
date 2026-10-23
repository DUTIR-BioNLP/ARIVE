"""Run the complete two-pass ARIVE dataset-construction pipeline.

The individual stage modules remain the source of truth.  This entry point
validates their shared inputs, invokes them in dependency order, checks every
intermediate JSON boundary, and supports safe resume after an interrupted run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from project_paths import (
    ANATOMY_SOURCE,
    BASE_RECORDS,
    CLINICAL_CONTEXT,
    FEW_SHOT_EXAMPLES,
    IMAGE_ROOT,
    MCOT_ANNOTATION,
    PIPELINE_WORK_DIR,
    REFERENCE_REPORTS,
)

from .box_utils import BOX_RE, OBJECT_REF_BOX_RE
from .common import dump_json, load_json_object, load_records, record_id, report_text
from .generate_reasoning import (
    DEFAULT_REASONING_PROMPT, REASONING_INSTRUCTION,
    format_few_shot_examples, validate_prompt_config,
)
from .prepare_anatomy_boxes import load_source as load_anatomy_source
from .recover_current_reports import load_overrides, looks_incomplete


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = 12


@dataclass(frozen=True)
class PipelinePaths:
    work_dir: Path
    recovered_base: Path
    report_recovery_audit: Path
    reasoning_shard: Path
    reasoning: Path
    normalized_anatomy: Path
    reasoning_with_boxes: Path
    missing_anatomy: Path
    bbox_shard: Path
    mcot: Path
    mcot_with_object_refs: Path
    manifest: Path

    @classmethod
    def under(cls, work_dir: Path) -> "PipelinePaths":
        work_dir = work_dir.resolve()
        return cls(
            work_dir=work_dir,
            recovered_base=work_dir / "base_records_recovered.json",
            report_recovery_audit=work_dir / "report_recovery_audit.json",
            reasoning_shard=work_dir / "reasoning" / "generated.json",
            reasoning=work_dir / "reasoning.json",
            normalized_anatomy=work_dir / "anatomy.json",
            reasoning_with_boxes=work_dir / "reasoning_with_boxes.json",
            missing_anatomy=work_dir / "missing_anatomy.json",
            bbox_shard=work_dir / "bbox" / "generated.json",
            mcot=work_dir / "mcot.json",
            mcot_with_object_refs=work_dir / "mcot_with_object_refs.json",
            manifest=work_dir / "pipeline_manifest.json",
        )

    def generated_files(self, final_output: Path) -> list[Path]:
        return [
            self.recovered_base,
            self.report_recovery_audit,
            self.reasoning_shard,
            self.reasoning,
            self.normalized_anatomy,
            self.reasoning_with_boxes,
            self.missing_anatomy,
            self.bbox_shard,
            self.mcot,
            self.mcot_with_object_refs,
            final_output.resolve(),
        ]


@dataclass(frozen=True)
class Stage:
    name: str
    output: Path
    command: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-records", type=Path, default=BASE_RECORDS)
    parser.add_argument("--report-archive", type=Path,
                        help="MIMIC-CXR report ZIP; use IMPRESSION when a current report is truncated")
    parser.add_argument("--report-overrides", type=Path,
                        help="JSON object mapping record IDs to manually verified current reports")
    parser.add_argument("--reference-reports", type=Path, default=REFERENCE_REPORTS)
    parser.add_argument("--clinical-context", type=Path, default=CLINICAL_CONTEXT)
    parser.add_argument("--few-shot-examples", type=Path, default=FEW_SHOT_EXAMPLES)
    parser.add_argument('--reasoning-prompt-config', type=Path,
                        help='Optional JSON instruction/developer_instruction for merged reasoning generation.')
    parser.add_argument("--draft-reasoning", type=Path,
                        help="Revise a saved reasoning shard instead of generating new drafts.")
    parser.add_argument("--review-notes", type=Path,
                        help="Optional per-ID evidence-audit notes for --evidence-review.")
    parser.add_argument("--skip-evidence-review", action="store_true")
    parser.add_argument("--evidence-review", dest="skip_evidence_review", action="store_false")
    parser.set_defaults(skip_evidence_review=True)
    parser.add_argument(
        "--few-shot-example-id",
        dest="few_shot_example_ids",
        action="append",
        default=[],
        help="Optional example id to select, in display order; repeat for multiple ids.",
    )
    parser.add_argument("--anatomy-source", type=Path, default=ANATOMY_SOURCE)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--model", required=True, help="Local path or model id for GPT-OSS.")
    parser.add_argument("--work-dir", type=Path, default=PIPELINE_WORK_DIR)
    parser.add_argument("--output", type=Path, default=MCOT_ANNOTATION)

    parser.add_argument("--start", type=int, default=0, help="Inclusive base-record offset.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive base-record offset.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reasoning-max-model-len", type=int, default=20480)
    parser.add_argument("--reasoning-max-tokens", type=int, default=10240)
    parser.add_argument("--bbox-max-model-len", type=int, default=30720)
    parser.add_argument("--bbox-max-tokens", type=int, default=30720)

    parser.add_argument(
        "--anatomy-is-normalized",
        action="store_true",
        help="Treat --anatomy-source as prepare_anatomy_boxes output.",
    )
    parser.add_argument("--keep-all-anatomy", action="store_true")
    parser.add_argument("--legacy-swapped-dimensions", action="store_true")
    parser.add_argument(
        "--allow-missing-anatomy",
        action="store_true",
        help="Allow records without an anatomy match to be omitted from the final dataset.",
    )
    parser.add_argument("--join-labels-with", default="; ")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed, validated stages from a matching prior invocation.",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute every stage and replace this pipeline's existing outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print stage commands without writing or running them.",
    )
    return parser.parse_args(argv)


def _append_option(command: list[str], name: str, value: Any) -> None:
    if value is not None:
        command.extend([name, str(value)])


def _module_command(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def build_stages(
    args: argparse.Namespace,
    paths: PipelinePaths,
    *,
    partial_selection: bool,
) -> list[Stage]:
    overwrite = ["--overwrite"] if args.overwrite else []
    base_path = paths.recovered_base if (args.report_archive or args.report_overrides) else args.base_records.resolve()
    stages: list[Stage] = []
    if args.report_archive or args.report_overrides:
        recover = _module_command("data_generation.recover_current_reports")
        recover.extend([
            "--input", str(args.base_records.resolve()),
            "--start", str(args.start),
            "--output", str(paths.recovered_base),
            "--audit-output", str(paths.report_recovery_audit),
        ])
        _append_option(recover, "--end", args.end)
        _append_option(recover, "--report-archive", args.report_archive.resolve() if args.report_archive else None)
        _append_option(recover, "--report-overrides", args.report_overrides.resolve() if args.report_overrides else None)
        recover.extend(overwrite)
        stages.append(Stage("recover_current_reports", paths.recovered_base, tuple(recover)))

    reasoning = _module_command("data_generation.generate_reasoning")
    reasoning.extend(
        [
            "--input",
            str(base_path),
            "--clinical-context",
            str(args.clinical_context.resolve()),
            "--reference-reports",
            str(args.reference_reports.resolve()),
            "--few-shot-examples",
            str(args.few_shot_examples.resolve()),
            "--model",
            args.model,
            "--start",
            str(args.start),
            "--batch-size",
            str(args.batch_size),
            "--max-model-len",
            str(args.reasoning_max_model_len),
            "--max-tokens",
            str(args.reasoning_max_tokens),
            "--temperature",
            str(args.temperature),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--output",
            str(paths.reasoning_shard),
        ]
    )
    _append_option(reasoning, "--end", args.end)
    _append_option(reasoning, "--seed", args.seed)
    _append_option(reasoning, '--prompt-config', args.reasoning_prompt_config.resolve() if args.reasoning_prompt_config else None)
    _append_option(reasoning, "--draft-reasoning", args.draft_reasoning)
    _append_option(reasoning, "--review-notes", args.review_notes)
    if args.skip_evidence_review:
        reasoning.append("--skip-evidence-review")
    else:
        reasoning.append("--evidence-review")
    for example_id in args.few_shot_example_ids:
        reasoning.extend(["--few-shot-example-id", str(example_id)])
    reasoning.extend(overwrite)

    merge = _module_command("data_generation.merge_reasoning")
    merge.extend(
        [
            "--base",
            str(base_path),
            "--reasoning",
            str(paths.reasoning_shard),
            "--output",
            str(paths.reasoning),
        ]
    )
    if partial_selection:
        merge.append("--allow-missing")
    merge.extend(overwrite)

    stages.extend([
        Stage("generate_reasoning", paths.reasoning_shard, tuple(reasoning)),
        Stage("merge_reasoning", paths.reasoning, tuple(merge)),
    ])

    if args.anatomy_is_normalized:
        anatomy_path = args.anatomy_source.resolve()
    else:
        prepare = _module_command("data_generation.prepare_anatomy_boxes")
        prepare.extend(
            [
                "--input",
                str(args.anatomy_source.resolve()),
                "--image-root",
                str(args.image_root.resolve()),
                "--output",
                str(paths.normalized_anatomy),
            ]
        )
        if args.keep_all_anatomy:
            prepare.append("--keep-all")
        if args.legacy_swapped_dimensions:
            prepare.append("--legacy-swapped-dimensions")
        prepare.extend(overwrite)
        stages.append(Stage("prepare_anatomy_boxes", paths.normalized_anatomy, tuple(prepare)))
        anatomy_path = paths.normalized_anatomy

    attach = _module_command("data_generation.attach_anatomy_boxes")
    attach.extend(
        [
            "--records",
            str(paths.reasoning),
            "--anatomy",
            str(anatomy_path),
            "--output",
            str(paths.reasoning_with_boxes),
            "--missing-output",
            str(paths.missing_anatomy),
            *overwrite,
        ]
    )
    stages.append(Stage("attach_anatomy_boxes", paths.reasoning_with_boxes, tuple(attach)))

    bbox = _module_command("data_generation.insert_bboxes")
    bbox.extend(
        [
            "--input",
            str(paths.reasoning_with_boxes),
            "--model",
            args.model,
            "--start",
            "0",
            "--batch-size",
            str(args.batch_size),
            "--max-model-len",
            str(args.bbox_max_model_len),
            "--max-tokens",
            str(args.bbox_max_tokens),
            "--temperature",
            str(args.temperature),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--output",
            str(paths.bbox_shard),
        ]
    )
    _append_option(bbox, "--seed", args.seed)
    bbox.extend(overwrite)
    stages.append(Stage("insert_bboxes", paths.bbox_shard, tuple(bbox)))

    build = _module_command("data_generation.build_mcot_dataset")
    build.extend(
        [
            "--bbox-source",
            str(paths.reasoning_with_boxes),
            "--predictions",
            str(paths.bbox_shard),
            "--validate-coordinates",
            "--preserve-source-text",
            "--restore-missing-conclusion",
            "--output",
            str(paths.mcot),
            *overwrite,
        ]
    )
    stages.append(Stage("build_mcot_dataset", paths.mcot, tuple(build)))

    object_refs = _module_command("data_generation.add_object_refs")
    object_refs.extend(
        [
            "--mcot",
            str(paths.mcot),
            "--bbx",
            str(paths.reasoning_with_boxes),
            "--join-labels-with",
            args.join_labels_with,
            "--output",
            str(paths.mcot_with_object_refs),
            *overwrite,
        ]
    )
    stages.append(
        Stage("add_object_refs", paths.mcot_with_object_refs, tuple(object_refs))
    )

    convert = _module_command("data_generation.convert_to_qwenvl")
    convert.extend(
        [
            "--mcot",
            str(paths.mcot_with_object_refs),
            "--clinical-context",
            str(args.clinical_context.resolve()),
            "--reference-reports",
            str(args.reference_reports.resolve()),
            "--output",
            str(args.output.resolve()),
            *overwrite,
        ]
    )
    stages.append(Stage("convert_to_qwenvl", args.output.resolve(), tuple(convert)))
    return stages


def _validate_positive_arguments(args: argparse.Namespace) -> None:
    if args.skip_evidence_review and (args.draft_reasoning or args.review_notes):
        raise ValueError("saved drafts and review notes require --evidence-review")
    if args.start < 0 or (args.end is not None and args.end < args.start):
        raise ValueError("invalid --start/--end range")
    for name in (
        "batch_size",
        "tensor_parallel_size",
        "reasoning_max_model_len",
        "reasoning_max_tokens",
        "bbox_max_model_len",
        "bbox_max_tokens",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")


def _validate_inputs(args: argparse.Namespace) -> tuple[int, bool]:
    _validate_positive_arguments(args)
    for name in (
        "base_records",
        "reference_reports",
        "clinical_context",
        "few_shot_examples",
        "anatomy_source",
    ):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"--{name.replace('_', '-')} is not a file: {path}")
    for name in ("report_archive", "report_overrides", "draft_reasoning", "review_notes", "reasoning_prompt_config"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"--{name.replace('_', '-')} is not a file: {path}")
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"--image-root is not a directory: {args.image_root}")

    records = load_records(args.base_records)
    if not records:
        raise ValueError("--base-records is empty")
    selected_end = min(args.end if args.end is not None else len(records), len(records))
    selected_count = max(0, selected_end - args.start)
    if selected_count == 0:
        raise ValueError("the selected --start/--end range is empty")
    overrides = load_overrides(args.report_overrides)
    selected = records[args.start:selected_end]
    selected_ids = {str(record_id(record)) for record in selected}
    unknown = set(overrides) - selected_ids
    if unknown:
        raise ValueError(f"report override IDs are not selected: {sorted(unknown)}")
    if args.report_archive is None:
        incomplete = [record_id(record) for record in selected
                      if str(record_id(record)) not in overrides
                      and looks_incomplete(report_text(record))]
        if incomplete:
            raise ValueError(
                f"{len(incomplete)} selected current reports are non-diagnostic "
                f"(first IDs: {incomplete[:10]}); supply --report-archive "
                "and/or --report-overrides"
            )

    # Parse all shared inputs before the expensive first model load. Raw
    # anatomy annotations may be CSV, while already-normalized annotations use
    # the regular JSON/JSONL record loader consumed by attach_anatomy_boxes.
    load_records(args.reference_reports)
    load_json_object(args.clinical_context)
    if args.reasoning_prompt_config:
        validate_prompt_config(load_json_object(args.reasoning_prompt_config))
    format_few_shot_examples(
        load_records(args.few_shot_examples),
        args.few_shot_example_ids,
    )
    if args.anatomy_is_normalized:
        load_records(args.anatomy_source)
    else:
        load_anatomy_source(args.anatomy_source)
    partial = args.start != 0 or selected_end != len(records)
    return selected_count, partial


def _validate_path_separation(
    args: argparse.Namespace,
    paths: PipelinePaths,
    stages: Sequence[Stage],
) -> None:
    inputs = {
        path.resolve()
        for path in (
            args.base_records,
            args.reference_reports,
            args.clinical_context,
            args.few_shot_examples,
            args.anatomy_source,
        )
    }
    inputs.update(path.resolve() for path in (args.report_archive, args.report_overrides, args.reasoning_prompt_config) if path)
    if args.reasoning_prompt_config is None:
        inputs.add(DEFAULT_REASONING_PROMPT.resolve())
    outputs = [paths.manifest, *(stage.output.resolve() for stage in stages)]
    if any(stage.name == "recover_current_reports" for stage in stages):
        outputs.append(paths.report_recovery_audit.resolve())
    if len(outputs) != len(set(outputs)):
        raise ValueError(
            "--output must not reuse an intermediate path inside --work-dir"
        )
    overlap = inputs.intersection(outputs)
    if overlap:
        targets = ", ".join(str(path) for path in sorted(overlap))
        raise ValueError(f"pipeline output paths must not overwrite inputs: {targets}")


def _manifest(args: argparse.Namespace, selected_count: int) -> dict[str, Any]:
    def input_identity(path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    model_path = Path(args.model).expanduser()
    model = str(model_path.resolve()) if model_path.exists() else args.model
    return {
        "pipeline_version": PIPELINE_VERSION,
        "base_records": input_identity(args.base_records),
        "report_archive": input_identity(args.report_archive) if args.report_archive else None,
        "report_overrides": input_identity(args.report_overrides) if args.report_overrides else None,
        "reference_reports": input_identity(args.reference_reports),
        "clinical_context": input_identity(args.clinical_context),
        "few_shot_examples": input_identity(args.few_shot_examples),
        "few_shot_example_ids": list(args.few_shot_example_ids),
        "reasoning_prompt_config": ({
            **input_identity(args.reasoning_prompt_config),
            'sha256': hashlib.sha256(args.reasoning_prompt_config.read_bytes()).hexdigest(),
        } if args.reasoning_prompt_config else None),
        "default_reasoning_prompt": ({
            **input_identity(DEFAULT_REASONING_PROMPT),
            'sha256': hashlib.sha256(DEFAULT_REASONING_PROMPT.read_bytes()).hexdigest(),
            'developer_instruction': REASONING_INSTRUCTION,
        } if args.reasoning_prompt_config is None else None),
        "draft_reasoning": input_identity(args.draft_reasoning) if args.draft_reasoning else None,
        "review_notes": input_identity(args.review_notes) if args.review_notes else None,
        "skip_evidence_review": args.skip_evidence_review,
        "anatomy_source": input_identity(args.anatomy_source),
        "anatomy_is_normalized": args.anatomy_is_normalized,
        "image_root": str(args.image_root.resolve()),
        "model": model,
        "output": str(args.output.resolve()),
        "start": args.start,
        "end": args.end,
        "selected_count": selected_count,
        "batch_size": args.batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "temperature": args.temperature,
        "seed": args.seed,
        "reasoning_max_model_len": args.reasoning_max_model_len,
        "reasoning_max_tokens": args.reasoning_max_tokens,
        "bbox_max_model_len": args.bbox_max_model_len,
        "bbox_max_tokens": args.bbox_max_tokens,
        "keep_all_anatomy": args.keep_all_anatomy,
        "legacy_swapped_dimensions": args.legacy_swapped_dimensions,
        "allow_missing_anatomy": args.allow_missing_anatomy,
        "join_labels_with": args.join_labels_with,
    }


def _prepare_work_dir(
    args: argparse.Namespace,
    paths: PipelinePaths,
    manifest: dict[str, Any],
) -> None:
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not paths.manifest.is_file():
            raise FileNotFoundError(
                f"cannot --resume without the pipeline manifest: {paths.manifest}"
            )
        with paths.manifest.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != manifest:
            raise ValueError(
                "resume arguments do not match pipeline_manifest.json; "
                "use the original arguments or start a new --work-dir"
            )
        return

    conflicts = [
        path for path in paths.generated_files(args.output) if path.exists()
    ]
    if conflicts and not args.overwrite:
        preview = "\n".join(f"  - {path}" for path in conflicts)
        raise FileExistsError(
            "pipeline outputs already exist; use --resume or --overwrite:\n" + preview
        )
    dump_json(paths.manifest, manifest, overwrite=args.overwrite)


def _runtime_environment() -> dict[str, str]:
    env = os.environ.copy()
    runtime_lib = Path(sys.prefix) / "lib"
    if (runtime_lib / "libstdc++.so.6").is_file():
        current = [part for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
        runtime_text = str(runtime_lib.resolve())
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [runtime_text, *(part for part in current if part != runtime_text)]
        )
    return env


def _validate_stage_output(stage: Stage) -> list[dict[str, Any]]:
    records = load_records(stage.output)
    if stage.name != "attach_anatomy_boxes" and not records:
        raise ValueError(f"stage {stage.name} produced no records: {stage.output}")
    # Every record-producing boundary must retain a stable identifier, except
    # normalized anatomy which is keyed by image path.
    if stage.name != "prepare_anatomy_boxes":
        for record in records:
            record_id(record)
    return records


def _validate_grounding(path: Path) -> None:
    for record in load_records(path):
        reasoning = record.get("reasoning")
        if not isinstance(reasoning, str):
            raise TypeError(f"grounded record id={record_id(record)} has no reasoning string")
        boxes = len(BOX_RE.findall(reasoning))
        refs = len(OBJECT_REF_BOX_RE.findall(reasoning))
        if boxes != refs:
            raise ValueError(
                f"grounded record id={record_id(record)} has boxes={boxes}, "
                f"adjacent object_refs={refs}"
            )


def _validate_final_annotations(path: Path) -> int:
    records = load_records(path)
    if not records:
        raise ValueError(f"final annotation is empty: {path}")
    for record in records:
        images = record.get("image")
        conversations = record.get("conversations")
        if not isinstance(images, list) or not all(isinstance(item, str) for item in images):
            raise TypeError(f"final record id={record_id(record)} has invalid image paths")
        if not isinstance(conversations, list):
            raise TypeError(f"final record id={record_id(record)} has invalid conversations")
        placeholders = sum(
            turn.get("value", "").count("<image>")
            for turn in conversations
            if isinstance(turn, dict) and isinstance(turn.get("value"), str)
        )
        if placeholders != len(images):
            raise ValueError(
                f"final record id={record_id(record)} has image paths={len(images)}, "
                f"placeholders={placeholders}"
            )
    return len(records)


def run_pipeline(args: argparse.Namespace) -> None:
    selected_count, partial_selection = _validate_inputs(args)
    paths = PipelinePaths.under(args.work_dir)
    stages = build_stages(args, paths, partial_selection=partial_selection)
    _validate_path_separation(args, paths, stages)

    print(f"[pipeline] selected base records: {selected_count}")
    for index, stage in enumerate(stages, start=1):
        print(f"[pipeline] {index}/{len(stages)} {stage.name}: {shlex.join(stage.command)}")
    if args.dry_run:
        print("[pipeline] dry run complete; nothing was written")
        return

    manifest = _manifest(args, selected_count)
    _prepare_work_dir(args, paths, manifest)
    env = _runtime_environment()
    runtime_lib = env.get("LD_LIBRARY_PATH", "").split(os.pathsep)[0]
    if runtime_lib:
        print(f"[pipeline] child runtime library path begins with: {runtime_lib}")

    upstream_recomputed = False
    for index, stage in enumerate(stages, start=1):
        if args.resume and not upstream_recomputed and stage.output.is_file():
            print(f"[pipeline] {index}/{len(stages)} resume {stage.name}: {stage.output}")
        else:
            print(f"[pipeline] {index}/{len(stages)} start {stage.name}", flush=True)
            command = list(stage.command)
            # If an upstream stage had to be reconstructed during --resume,
            # every downstream artifact is stale and must be replaced too.
            if stage.output.exists() and "--overwrite" not in command:
                command.append("--overwrite")
            try:
                subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"pipeline stage {stage.name} failed with exit code {exc.returncode}; "
                    "fix the reported cause and rerun the same command with --resume"
                ) from exc
            upstream_recomputed = True
        records = _validate_stage_output(stage)
        print(
            f"[pipeline] {index}/{len(stages)} complete {stage.name}: "
            f"records={len(records)}",
            flush=True,
        )

        if stage.name == "attach_anatomy_boxes":
            missing = load_records(paths.missing_anatomy)
            if missing and not args.allow_missing_anatomy:
                raise ValueError(
                    f"{len(missing)} selected records have no anatomy match; "
                    f"inspect {paths.missing_anatomy} or pass --allow-missing-anatomy"
                )
        elif stage.name == "add_object_refs":
            _validate_grounding(stage.output)

    final_count = _validate_final_annotations(args.output.resolve())
    print(
        f"[pipeline] complete: annotations={final_count}, output={args.output.resolve()}, "
        f"work_dir={paths.work_dir}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    run_pipeline(parse_args(argv))


if __name__ == "__main__":
    main()
