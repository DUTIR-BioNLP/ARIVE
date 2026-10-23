# ARIVE： **Interleaving anatomical inspection and diagnostic reasoning for explainable radiology report generation**.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.12-3776AB?logo=python&logoColor=white" alt="Python 3.10 and 3.12">
  <img src="https://img.shields.io/badge/Base-Qwen3--VL--4B-7B61FF" alt="Qwen3-VL-4B">
  <img src="https://img.shields.io/badge/License-Apache--2.0-2F80ED" alt="Apache-2.0">
  <img src="https://img.shields.io/badge/Status-Research%20Only-B31B1B" alt="Research use only">
</p>

ARIVE uses Qwen3-VL-4B for grounded chest X-ray report generation. The model
emits anatomical regions and boxes, receives crop observations from a local
vision branch, and continues reasoning. The repository includes data
construction, SFT, GRPO, and inference.

[Hugging Face model](https://huggingface.co/akinaHCT/ARIVE) · [中文说明](README_zh-CN.md) · [Data format](docs/DATA_FORMAT.md) · [Code provenance](docs/PROVENANCE.md)

> Research use only; not for clinical diagnosis or patient care.

## Contents

- [1. Setup](#1-setup)
- [2. Inference](#2-inference)
- [3. Data construction](#3-data-construction)
  - [Inputs and environment](#inputs-and-environment)
  - [3.0 Replay a sample](#30-replay-one-synthetic-or-local-real-record)
  - [3.1 Build the dataset](#31-build-the-dataset)
    - [3.1.1 Complete pipeline](#311-run-the-complete-pipeline-with-one-command)
    - [3.1.2 Generate reasoning](#312-generate-reasoning-with-few-shot-style-clinical-context-and-prior-findings)
    - [3.1.3 Normalize and attach anatomy boxes](#313-normalize-and-attach-anatomy-boxes)
    - [3.1.4 Insert boxes and object references](#314-insert-boxes-and-object-references)
    - [3.1.5 Convert to Qwen-VL conversations](#315-convert-to-qwen-vl-conversations)
    - [3.1.6 Worked example](#316-worked-example-one-annotation-through-data_generation)
  - [3.2 Training annotation](#32-training-annotation)
- [4. Training](#4-training)
- [Checks and license](#checks-and-license)

## 1. Setup

Use Linux with NVIDIA GPUs. The reference environments use Python 3.12 for data
generation and Python 3.10 for training/inference. Install training dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/training.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

Copy `configs/env.example` to `.env`, edit the data/model paths, and load it below.
Leave `QWENVL_ENV_DIR` unset when using the active environment. All commands run
from the repository root.

```bash
set -a
source .env
set +a
```

Code lives in `data_generation/` (construction), `qwenvl/` (training/rollout),
`inference/`, and `scripts/` / `tools/` (launchers and utilities).
Existing `mcot` filenames and `MCOT_*` settings are retained for compatibility.

## 2. Inference

We provide the final trained [ARIVE checkpoint](https://huggingface.co/akinaHCT/ARIVE),
containing the decoder and both global and local vision branches.

For the current inference code, obtain the Qwen3-VL-4B-Instruct base model
separately and create the vLLM export locally from the released checkpoint:

```bash
export QWENVL_INFER_BASE_MODEL=/path/to/Qwen3-VL-4B-Instruct
export QWENVL_INFER_VISION_CHECKPOINT=/path/to/ARIVE
export QWENVL_INFER_VLLM_CHECKPOINT=/path/to/ARIVE-vllm

python tools/export_vllm_checkpoint.py \
  --src "$QWENVL_INFER_VISION_CHECKPOINT" \
  --dst "$QWENVL_INFER_VLLM_CHECKPOINT"
```

The export directory must be absent or empty. Keep the released checkpoint's
tokenizer/processor files and chat template (`chat_template.jinja` or
`chat_template.json`) with assistant-side media support; the exporter copies
them into the vLLM directory.

For MIMIC-CXR inference, download `libra_findings_section_eval.jsonl` directly
from [Libra's Data Download section](https://github.com/X-iZhang/Libra#data-download).
If the file uses Libra's `question_id` and `reference` fields, map them to `id`
and `findings`. Save it as `data/inference/evaluation_reports.jsonl`, or point
`QWENVL_INFER_REPORT_PATH` to the prepared file.

Prepare `data/inference/{evaluation_reports.jsonl,reference_reports.jsonl,clinical_context.json}`
and the image tree at `data/images/`, or set the corresponding `QWENVL_INFER_*`
paths. Evaluation rows contain `id`, `image` (current first, optional prior
second), and `findings` or `report`. Reference reports cover all prior studies;
clinical context is keyed by current study ID and supplies `Indication`.
See [data formats](docs/DATA_FORMAT.md) for examples.

```bash
CUDA_VISIBLE_DEVICES=0 python inference/infer_mcot.py
```

Results are written as a JSON list under `QWENVL_INFER_OUTPUT_DIR`. Defaults:
temperature `0.7`, top-p `0.8`, top-k `-1`, presence penalty `1.5`. Existing
results require `QWENVL_INFER_OVERWRITE=1` to replace.

For MIMIC-ABN, configure the two Prior-RRG annotation paths in `.env`: the ABN
annotation selects test members, and the full annotation supplies prior findings.
Then run the abnormal-only entry point:

```bash
export QWENVL_INFER_REPORT_PATH="$MCOT_DATA_ROOT/inference/mimic_abn_evaluation_reports.jsonl"
export QWENVL_INFER_OUTPUT_DIR="$PWD/outputs/inference_abn"
QWENVL_INFER_PREFLIGHT_ONLY=1 python inference/infer_mcot_abn.py
CUDA_VISIBLE_DEVICES=0 python inference/infer_mcot_abn.py
```

## 3. Data construction

The implementation is in `data_generation/`. By default, two GPT-OSS passes
generate report/context/prior-aware reasoning and select its spatial anchors.
`--evidence-review` adds an optional revision pass using the same loaded model.
The remaining stages join, normalize, validate, and convert records.

### Inputs and environment

Prepare reports, curated few-shot examples, clinical context, anatomy boxes,
and the corresponding images. Real data and model weights are not bundled.
The default construction layout is:

```text
data/
├── source/
│   ├── base_records.json
│   ├── few_shot_examples.json
│   ├── reference_reports.json
│   ├── clinical_context.json
│   └── anatomy_boxes.json
├── images/                       # image tree or symlink
├── work/pipeline/                # intermediate outputs
├── mcot_annotation.json          # final training annotation
├── example_synthetic/            # public synthetic fixture
└── sample/                       # ignored local real sample
```

`MCOT_DATA_ROOT` overrides the data root; individual path arguments take
precedence. See [data/README.md](data/README.md) for the complete layout.
Create private input directories and link the image tree:

```bash
mkdir -p data/source data/inference data/work/pipeline
ln -s /path/to/MIMIC-CXR-JPG data/images
```

Install GPT-OSS generation dependencies in a separate environment:

```bash
python -m venv .venv-generation
source .venv-generation/bin/activate
python -m pip install -r requirements/generation.txt
```

### 3.0 Replay one synthetic or local real record

The repository includes mutually related synthetic JSON files, two 8×8 PGM
placeholder images, precomputed outputs for both GPT-OSS calls, and the expected
final annotation. It contains no patient information and directly validates
ID/study/path joins, pixel-coordinate normalization, bbox merging, object
references, and Qwen-VL conversion:

```bash
bash scripts/run_sample_data_pipeline.sh
```

The default command reads `data/example_synthetic/`, writes its intermediates
under `work/`, and compares the result with `expected/mcot_annotation.json`.
It replays the bundled Stage 1 and Stage 5 outputs, so GPT-OSS is not required.
See [data/example_synthetic/README.md](data/example_synthetic/README.md) for the
full-pipeline and `--dry-run` commands using those inputs.

For an authorized local dataset, extract one internally linked real record into
the gitignored `data/sample/` tree. The utility selects the current/prior report,
clinical context, raw anatomy row, real few-shot examples, and saved outputs from
both generation passes; it copies only the selected current/prior images from
the local image root. Historical local outputs used the legacy swapped-dimension
normalization, so the extractor adds a private marker that makes the replay
script enable that compatibility flag automatically:

```bash
python tools/build_local_real_sample.py \
  --record-id RECORD_ID \
  --report-corpus /private/path/report_corpus.json \
  --clinical-context /private/path/clinical_context.json \
  --few-shot-examples /private/path/few_shot_examples.jsonl \
  --anatomy-source /private/path/anatomy_source.json \
  --reasoning-predictions /private/path/reasoning_predictions.json \
  --bbox-predictions /private/path/bbox_predictions.json \
  --image-root /private/path/MIMIC-CXR

MCOT_SAMPLE_ROOT="$PWD/data/sample" bash scripts/run_sample_data_pipeline.sh
```

The command writes real-case-derived JSON under `data/sample/work/` and
`data/sample/output/`. After reviewing the result, an optional regression
snapshot can be recorded locally with:

```bash
mkdir -p data/sample/expected
cp data/sample/output/mcot_annotation.json data/sample/expected/mcot_annotation.json
```

Do not commit or redistribute any part of `data/sample/`.

### 3.1 Build the dataset

Run all modules from the repository root with `python -m`. Complete schemas and
path conventions are documented in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).
The commands below show one reproducible sequence; choose shard boundaries for
your hardware.

Define inputs and a work directory:

```bash
export GPT_OSS_MODEL=/path/to/gpt-oss-120b
export MCOT_DATA_ROOT="$PWD/data"
export BASE_RECORDS="$MCOT_DATA_ROOT/source/base_records.json"
export FEW_SHOT_EXAMPLES="$MCOT_DATA_ROOT/source/few_shot_examples.json"
export REFERENCE_REPORTS="$MCOT_DATA_ROOT/source/reference_reports.json"
export CLINICAL_CONTEXT="$MCOT_DATA_ROOT/source/clinical_context.json"
export ANATOMY_SOURCE="$MCOT_DATA_ROOT/source/anatomy_boxes.json"
export MIMIC_CXR_ROOT="$MCOT_DATA_ROOT/images"
export PIPELINE_WORK_DIR="$MCOT_DATA_ROOT/work/pipeline"
mkdir -p "$PIPELINE_WORK_DIR/reasoning" "$PIPELINE_WORK_DIR/bbox"
```

#### 3.1.1 Run the complete pipeline with one command

The integrated entry point runs both GPT-OSS passes and every deterministic
conversion stage in the order documented below:

If a current report is obviously truncated (for example, only `At the` or an
acquisition timestamp), the pipeline now rejects it before model generation.
To recover these cases from the original report archive, add
`--report-archive /path/to/mimic-cxr-reports.zip`. The pipeline uses the
corresponding study's `IMPRESSION` as the current report and records every
replacement in `work/pipeline/report_recovery_audit.json`. For ambiguous or
contradictory impressions, supply a reviewed JSON object such as
`{"11011839": "Verified current report text..."}` through
`--report-overrides /path/to/overrides.json`; overrides take precedence.
This adds a pre-generation recovery stage without altering the source file.
The bbox merge stage also restores a final summary paragraph if the bbox-only
model pass drops it.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m data_generation.run_pipeline \
  --model "$GPT_OSS_MODEL" \
  --tensor-parallel-size 8
```

This short command uses the `data/` defaults in `project_paths.py`. Explicit
`--base-records`, `--image-root`, `--work-dir`, and `--output` arguments remain
available when any file lives elsewhere.

The table below is the exact data-file contract for every construction stage.
Intermediate paths without a leading variable are relative to `--work-dir`;
the Stage 8 destination is the path passed to `--output`. In a manually sharded
run, replace `generated.json` with all non-overlapping shard files.

| Stage | Data files/directories read | Core operation | Files written | External compute/image access |
| --- | --- | --- | --- | --- |
| 1 · `generate_reasoning` | `$BASE_RECORDS` (JSON list or JSONL), `$FEW_SHOT_EXAMPLES` (JSON list or JSONL), `$CLINICAL_CONTEXT` (one JSON object), `$REFERENCE_REPORTS` (JSON list or JSONL) | Generate reasoning from few-shot examples, current/prior reports, and clinical context; optionally revise evidence fidelity | `reasoning/generated.json`, `reasoning/drafts/generated.json` | One GPT-OSS pass by default; no pixels, image paths only identify studies |
| 2 · `merge_reasoning` | `$BASE_RECORDS` and one or more Stage 1 reasoning JSON/JSONL shards | Join reasoning by `id` and enforce duplicate, missing, and overwrite rules | `reasoning.json` | None |
| 3 · `prepare_anatomy_boxes` | `$ANATOMY_SOURCE` (raw CSV, JSON list, or JSONL) and each referenced JPG below `$MIMIC_CXR_ROOT` | Read image dimensions and normalize source boxes to integer `[0,1000]` coordinates | `anatomy.json` | Opens JPGs only for width and height |
| 4 · `attach_anatomy_boxes` | `reasoning.json` and `anatomy.json`; with `--anatomy-is-normalized`, the normalized `$ANATOMY_SOURCE` JSON/JSONL replaces `anatomy.json` | Join anatomy boxes by normalized current-image path and record unmatched items separately | `reasoning_with_boxes.json` and `missing_anatomy.json` | None |
| 5 · `insert_bboxes` | `reasoning_with_boxes.json` | Select supplied region and sentence IDs; code inserts the source coordinates without rewriting the narrative | `bbox/generated.json`, `bbox/raw/generated.json` | Second GPT-OSS stage; no pixels |
| 6 · `build_mcot_dataset` | `reasoning_with_boxes.json` and one or more Stage 5 bbox-prediction JSON/JSONL shards | Join bbox predictions by `id` and validate the final ARIVE reasoning | `mcot.json` | None |
| 7 · `add_object_refs` | `mcot.json` plus `reasoning_with_boxes.json` as the ID/coordinate/label lookup | Add an adjacent anatomy object-reference tag before each matched box | `mcot_with_object_refs.json` | None |
| 8 · `convert_to_qwenvl` | `mcot_with_object_refs.json`, `$CLINICAL_CONTEXT`, and `$REFERENCE_REPORTS` | Convert to Qwen-VL conversations and emit `<image>` plus virtual-crop paths in prior/current/crop order | Final `mcot_annotation.json` | No pixels; writes relative full-image and virtual-crop paths only |

This distinction is important: neither GPT-OSS call sees image pixels. Stage 3
is the only construction stage that opens the JPG files, and it reads them only
for their dimensions. The actual pixels are first consumed by SFT/GRPO or
inference. All paths stored in records should therefore resolve as
`$MIMIC_CXR_ROOT/<record image path>`. Stage 1/2/5/6 shards are joined by `id`;
clinical context and prior reports are resolved by the `s<digits>` study ID in
the image path; anatomy is joined by the normalized current-image path.

For a model that fits on one GPU, use one visible device and
`--tensor-parallel-size 1`. The command validates every input before loading
GPT-OSS, writes a manifest to the work directory, checks every intermediate
JSON file, rejects missing anatomy by default, validates normalized boxes and
object-reference adjacency, and finally verifies that image paths and
`<image>` placeholders have identical counts.

An interrupted run can continue without repeating completed stages:

```bash
# Use exactly the same paths and generation arguments as the original command.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m data_generation.run_pipeline \
  --model "$GPT_OSS_MODEL" \
  --tensor-parallel-size 8 \
  --resume
```

`--resume` refuses changed arguments so stale intermediates cannot silently be
mixed into a new dataset. Use a new work directory for a different experiment,
or `--overwrite` to deliberately recompute every stage. Other useful controls
include `--start/--end` for a subset, `--anatomy-is-normalized` when the anatomy
input already has the normalized schema, and `--allow-missing-anatomy` when
dropped unmatched records are intentional. Run with `--dry-run` to validate
inputs and print all eight stage commands without writing files or loading the
model.

The work directory contains the following auditable boundaries:

```text
reasoning/generated.json
reasoning.json
anatomy.json
reasoning_with_boxes.json
missing_anatomy.json
bbox/generated.json
mcot.json
mcot_with_object_refs.json
pipeline_manifest.json
```

The separate commands in Sections 3.1.2--3.1.5 remain available when generation
must be scheduled as independently managed shards.

#### 3.1.2 Generate reasoning with few-shot style, clinical context, and prior findings

`BASE_RECORDS` must contain `id`, current/prior `image` paths, `report` or
`findings`; it no longer needs a pre-generated `reasoning`. `FEW_SHOT_EXAMPLES`
must contain curated records with readable reasoning narratives. Use the
repeatable `--few-shot-example-id` option when selecting examples from a larger
file. Select two or three complete, grounded walkthroughs with the depth expected
in the output, typically 350–500 words. Shorter instructions should not mean
shorter reasoning examples: a brief report paraphrase teaches the wrong task.
An example may include `report` (or `findings`) alongside its narrative, showing
how observations support interpretations, differentials, and uncertainty without
inventing unreported normal findings. The brief synthetic records in
`data/example_synthetic/inputs/grounded_few_shot_examples.json` illustrate the
input schema; they are not quality examples for detailed ARIVE construction.
For complete demonstrations, use
`--few-shot-examples data/example_synthetic/inputs/detailed_few_shot_examples.json`.
Paired examples may also include `clinical_context` (text) and `prior_report`.
Narrative-only legacy examples remain supported.
`REFERENCE_REPORTS` must contain every prior study referenced by the
input records.

Reasoning generation uses nonempty `Indication`, `History`, `Technique`, and
`Comparison`. The final Qwen-VL user turn includes only `Indication`, matching
inference; the other fields are used only during reasoning construction.

The default prompt is `data_generation/prompts/merged_reasoning.txt`. It requests
a detailed narrative with no fixed word count. To customize it, pass
`--reasoning-prompt-config /path/to/prompt.json` to `run_pipeline` or
`--prompt-config /path/to/prompt.json` to `generate_reasoning`. The JSON must
contain nonempty `instruction` and `developer_instruction` strings; complete
few-shot examples and current/context/prior inputs are retained. The manifest
records the prompt content hash, so changed prompts cannot reuse cached stages.
Format and length checks do not establish annotation quality.

Raw drafts are saved under `reasoning/drafts/`. Use
`--evidence-review --draft-reasoning /path/to/saved.json` to revise cached drafts
without generating them again. Revision remains an experimental text generation
pass: it can retain unsupported details and does not establish pixel-level correctness.
It selects sentence replacements while retaining unselected sentences and
paragraphs. Raw edits are saved under `reasoning/review/`. Optional
`--review-notes notes.json` provides a JSON object mapping IDs to case-specific
audit notes; these are review inputs, not reference reasoning. A depth check
rejects revisions that compress the walkthrough.

```bash
python -m data_generation.generate_reasoning \
  --input "$BASE_RECORDS" \
  --clinical-context "$CLINICAL_CONTEXT" \
  --reference-reports "$REFERENCE_REPORTS" \
  --few-shot-examples "$FEW_SHOT_EXAMPLES" \
  --model "$GPT_OSS_MODEL" \
  --start 0 \
  --end 10000 \
  --tensor-parallel-size 8 \
  --output "$PIPELINE_WORK_DIR/reasoning/000000-010000.json"
```

Repeat for all non-overlapping ranges, then join by stable record ID:

```bash
python -m data_generation.merge_reasoning \
  --base "$BASE_RECORDS" \
  --reasoning "$PIPELINE_WORK_DIR"/reasoning/*.json \
  --output "$PIPELINE_WORK_DIR/reasoning.json"
```

#### 3.1.3 Normalize and attach anatomy boxes

Skip `prepare_anatomy_boxes` if the boxes already follow the normalized output
schema. The default uses the standard x/width and y/height convention.
`--legacy-swapped-dimensions` exists only to reproduce historical artifacts.

```bash
python -m data_generation.prepare_anatomy_boxes \
  --input "$ANATOMY_SOURCE" \
  --image-root "$MIMIC_CXR_ROOT" \
  --output "$PIPELINE_WORK_DIR/anatomy.json"

python -m data_generation.attach_anatomy_boxes \
  --records "$PIPELINE_WORK_DIR/reasoning.json" \
  --anatomy "$PIPELINE_WORK_DIR/anatomy.json" \
  --output "$PIPELINE_WORK_DIR/reasoning_with_boxes.json" \
  --missing-output "$PIPELINE_WORK_DIR/missing_anatomy.json"
```

By default, an anatomy region is kept when its phrase-exists or abnormal flag
is true. Pass `--keep-all` only if every source region should be made available
to the spatial grounding prompt.

#### 3.1.4 Insert boxes and object references

The final GPT-OSS pass selects supplied region IDs and numbered sentence IDs.
Code validates those IDs and inserts the source coordinates
without copying or rewriting the narrative. Raw selections are saved under
`bbox/raw/`; the usual bbox shard still contains the complete annotated text.
It uses the same generation/sharding options as the first pass.

```bash
python -m data_generation.insert_bboxes \
  --input "$PIPELINE_WORK_DIR/reasoning_with_boxes.json" \
  --model "$GPT_OSS_MODEL" \
  --start 0 \
  --end 10000 \
  --tensor-parallel-size 8 \
  --output "$PIPELINE_WORK_DIR/bbox/000000-010000.json"

python -m data_generation.build_mcot_dataset \
  --bbox-source "$PIPELINE_WORK_DIR/reasoning_with_boxes.json" \
  --predictions "$PIPELINE_WORK_DIR"/bbox/*.json \
  --validate-coordinates \
  --preserve-source-text \
  --output "$PIPELINE_WORK_DIR/mcot.json"

python -m data_generation.add_object_refs \
  --mcot "$PIPELINE_WORK_DIR/mcot.json" \
  --bbx "$PIPELINE_WORK_DIR/reasoning_with_boxes.json" \
  --output "$PIPELINE_WORK_DIR/mcot_with_object_refs.json"
```

The executable order is `insert_bboxes` → `build_mcot_dataset` →
`add_object_refs`. The integrated pipeline enables `--preserve-source-text`:
box placements are transferred to the original narrative, and unsafe alignments
or coordinates absent from the supplied anatomy are rejected. For legacy
predictions without `grounding_method=source_sentence_anchors`, explicitly named
lung zones can also be reanchored to their own observation sentences. Current
sentence-ID selections retain their validated placements. `add_object_refs`
requires existing box tags and cannot run before box insertion.

Inspect the object-reference matching statistics. Fuzzy or ambiguous matches
should be reviewed before training. Correction shards can follow original
shards and replace the same ID only when `--allow-overwrite` is explicitly
passed to the merge/build command.

#### 3.1.5 Convert to Qwen-VL conversations

```bash
python -m data_generation.convert_to_qwenvl \
  --mcot "$PIPELINE_WORK_DIR/mcot_with_object_refs.json" \
  --clinical-context "$CLINICAL_CONTEXT" \
  --reference-reports "$REFERENCE_REPORTS" \
  --output "$MCOT_DATA_ROOT/mcot_annotation.json"
```

Crop entries in the annotation are virtual paths such as
`image.jpg_120_80_900_940`. They must not be materialized. The data loader
parses the suffix and crops the anchor image on demand.

#### 3.1.6 Worked example: one annotation through `data_generation`

The following synthetic example shows the exact record shape at each boundary.
Assume the current image is 1000 x 800 pixels. The source annotation contains a
current image, a prior image, and the target report. The first GPT-OSS call
generates the reasoning directly from this report and the curated few-shot examples:

```json
{
  "id": 1001,
  "image": [
    "files/p10/p100/s100/current.jpg",
    "files/p10/p100/s099/prior.jpg"
  ],
  "report": "Mild right basilar opacity, increased from the prior study."
}
```

A minimal few-shot file is a JSON list (or equivalent JSONL records) whose
entries expose a narrative through `reasoning`, `text`, or an assistant message:

```json
[
  {
    "id": "style-1",
    "reasoning": "We'll begin with image quality. Next I inspect the lungs, pleura, heart, and bones in order."
  }
]
```

The associated clinical-context entry and prior-report lookup are:

```json
{
  "s100": {
    "Indication": "Dyspnea",
    "History": "Shortness of breath",
    "Technique": "Portable AP",
    "Comparison": "Prior study"
  }
}
```

```json
{
  "id": 999,
  "image": ["files/p10/p100/s099/prior.jpg"],
  "findings": "No focal right basilar opacity."
}
```

1. `generate_reasoning` writes an ID-keyed shard. Large runs create multiple
   files with this same schema:

   ```json
   {
     "id": 1001,
     "reasoning": "Compared with the prior study, I see a new mild opacity in the right lower lung zone."
   }
   ```

2. `merge_reasoning` joins the shard back to the annotation metadata:

   ```json
   {
     "id": 1001,
     "image": [
       "files/p10/p100/s100/current.jpg",
       "files/p10/p100/s099/prior.jpg"
     ],
     "report": "Mild right basilar opacity, increased from the prior study.",
     "reasoning": "Compared with the prior study, I see a new mild opacity in the right lower lung zone."
   }
   ```

3. Given this raw anatomy annotation:

   ```json
   {
     "image": "files/p10/p100/s100/current.jpg",
     "bbox_coordinates": [[120, 160, 600, 720]],
     "bbox_labels": ["right lower lung zone"],
     "bbox_phrase_exists": [true],
     "bbox_is_abnormal": [true]
   }
   ```

   `prepare_anatomy_boxes` converts the pixel box to `[0, 1000]` coordinates:

   ```json
   {
     "image": "files/p10/p100/s100/current.jpg",
     "disease": [
       {
         "bbox_labels": "right lower lung zone",
         "bbox_coordinates": [120, 200, 600, 900]
       }
     ]
   }
   ```

4. `attach_anatomy_boxes` joins Stage 2 and Stage 3 by the current image path,
   adds the `disease` list, and writes unmatched full records separately to
   `missing_anatomy.json`:

   ```json
   {
     "id": 1001,
     "image": [
       "files/p10/p100/s100/current.jpg",
       "files/p10/p100/s099/prior.jpg"
     ],
     "report": "Mild right basilar opacity, increased from the prior study.",
     "reasoning": "Compared with the prior study, I see a new mild opacity in the right lower lung zone.",
     "disease": [
       {
         "bbox_labels": "right lower lung zone",
         "bbox_coordinates": [120, 200, 600, 900]
       }
     ]
   }
   ```

   When every current image matches, `missing_anatomy.json` is `[]`.

5. `insert_bboxes` asks GPT-OSS for region/sentence IDs, for example
   `[{"region_id": 0, "sentence_id": 0}]`. Code inserts the supplied box before
   that sentence and writes the shard record below; raw selections are saved
   separately in `bbox/raw/`:

   ```json
   {
     "id": 1001,
     "grounding_method": "source_sentence_anchors",
     "reasoning": "<|box_start|>120 200 600 900<|box_end|>\nCompared with the prior study, I see a new mild opacity in the right lower lung zone."
   }
   ```

6. `build_mcot_dataset` merges the shard, repairs/deduplicates box tags, and
   writes `mcot.json`:

   ```json
   {
     "id": 1001,
     "image": [
       "files/p10/p100/s100/current.jpg",
       "files/p10/p100/s099/prior.jpg"
     ],
     "report": "Mild right basilar opacity, increased from the prior study.",
     "reasoning": "<|box_start|>120 200 600 900<|box_end|>\nCompared with the prior study, I see a new mild opacity in the right lower lung zone."
   }
   ```

7. `add_object_refs` matches the box back
   to its anatomy entry:

   ```json
   {
     "id": 1001,
     "image": [
       "files/p10/p100/s100/current.jpg",
       "files/p10/p100/s099/prior.jpg"
     ],
     "report": "Mild right basilar opacity, increased from the prior study.",
     "reasoning": "<|object_ref_start|>right lower lung zone<|object_ref_end|><|box_start|>120 200 600 900<|box_end|>\nCompared with the prior study, I see a new mild opacity in the right lower lung zone."
   }
   ```

8. Finally, `convert_to_qwenvl` emits the annotation consumed by SFT. Notice
   that prior comes first, current comes second, and the last entry is a virtual
   crop of the current image:

   ```json
   {
     "id": 1001,
     "image": [
       "files/p10/p100/s099/prior.jpg",
       "files/p10/p100/s100/current.jpg",
       "files/p10/p100/s100/current.jpg_120_200_600_900"
     ],
     "conversations": [
       {
         "from": "human",
         "value": "...Prior chest X-ray:\n<image>\n...Current chest X-ray:\n<image>\n..."
       },
       {
         "from": "gpt",
         "value": "<think>\n<|object_ref_start|>right lower lung zone<|object_ref_end|><|box_start|>120 200 600 900<|box_end|>\n<image>\n\nCompared with the prior study, I see a new mild opacity in the right lower lung zone.\n</think>\nMild right basilar opacity, increased from the prior study."
       }
     ]
   }
   ```

Only the two shard-producing stages above call GPT-OSS. Every other transition
is deterministic data joining, normalization, validation, or schema conversion.

### 3.2 Training annotation

Stage 8 produces `mcot_annotation.json`: the human turn contains `Indication`,
current/prior images, and available prior findings; the assistant turn contains
grounded reasoning, crop placeholders, and the target report. SFT/GRPO read this
annotation and the images, without reopening the clinical-context or report files.

The default GRPO dataset is `ABN_MIMIC_COT`. Prepare
`abnormal_mcot_annotation.json` separately with the same schema, or set
`MCOT_GRPO_DATASET_NAME=MIMIC_COT` to use the Stage 8 output directly.

To update annotations made by the earlier converter, rerun Stage 8 from
`mcot_with_object_refs.json` into a new output file. Existing annotations are
not rewritten automatically. The pipeline version has changed, so old manifests
cannot be reused with `--resume`.

## 4. Training

Return to the training environment (`source .venv/bin/activate`) and load `.env`.
Set `MCOT_BASE_MODEL`, annotation/image paths, and checkpoint paths there.
Verify token IDs, run SFT, and export its full checkpoint:

```bash
python tools/verify_token_ids.py "$MCOT_BASE_MODEL"
CUDA_VISIBLE_DEVICES=0,1,2,3 MCOT_SFT_PREFLIGHT_ONLY=1 bash scripts/sft_qwen3_4b.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/sft_qwen3_4b.sh
python tools/export_vllm_checkpoint.py \
  --src "$MCOT_SFT_OUTPUT_DIR" --dst "$MCOT_VLLM_MODEL_PATH"
```

For GRPO, point `MCOT_SFT_MODEL_PATH` to the full SFT checkpoint and provide
answer/reasoning CheXbert weights and RadGraph assets through `.env`.
The launcher requires one node with at least four visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MCOT_GRPO_PREFLIGHT_ONLY=1 bash scripts/grpo_qwen3_4b_verl.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/grpo_qwen3_4b_verl.sh
```

Rewards are bbox IoU, reasoning CheXbert, answer CheXbert, and answer RadGraph;
weights are configured by `QWENVL_GRPO_REWARD_WEIGHTS`. SFT automatically resumes
existing `checkpoint-*`; GRPO requires `MCOT_GRPO_RESUME_FROM_CHECKPOINT`.
For inference, export the chosen full GRPO checkpoint with the same export tool
and use both matching directories. Export destinations must be absent or empty.

## Checks and license

```bash
python -m unittest discover -s tests -v
python -m compileall -q data_generation qwenvl inference tools
bash -n scripts/sft_qwen3_4b.sh
bash -n scripts/grpo_qwen3_4b_verl.sh
bash -n scripts/run_sample_data_pipeline.sh
```

Code: [Apache-2.0](LICENSE); attribution: [NOTICE](NOTICE). External models and
datasets retain their own terms. Keep restricted data and weights out of git.
