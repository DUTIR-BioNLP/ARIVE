# Data formats

ARIVE keeps image paths relative to `MIMIC_CXR_ROOT`. All examples below use
synthetic identifiers and text. Do not commit MIMIC-CXR images, reports,
clinical context, or generated patient-level annotations.

The default repository layout places construction inputs in `data/source/`,
images (or a symlink) in `data/images/`, inference inputs in `data/inference/`,
and the final annotation at `data/mcot_annotation.json`. Set `MCOT_DATA_ROOT`
to relocate this entire tree. A runnable public synthetic record is available
under `data/example_synthetic/`. A locally extracted real case may be placed in
the gitignored `data/sample/` directory.

## Report corpus

The reference-report lookup and inference stages accept JSONL records such as:

```json
{
  "id": 1001,
  "image": [
    "files/p10/p100/s100/current.jpg",
    "files/p10/p100/s099/prior.jpg"
  ],
  "findings": "Reference findings text."
}
```

`report` may replace `findings`. The first image is the current study. The
second is the prior study; omit it or repeat the current image when no prior is
available. Study IDs must appear as an `s` followed by digits in the path.

The first GPT-OSS pass takes base records with the same fields; a pre-generated
`reasoning` string is not required. It generates the final narrative directly
from the current report, curated few-shot reasoning examples, clinical context,
and the prior report when present.

## Few-shot reasoning examples

`--few-shot-examples` accepts a JSON list or JSONL file of curated records. Each
record must expose a reasoning narrative through `reasoning`, `text`, or an
assistant entry in `messages`. Use the repeatable `--few-shot-example-id` option
to select and order examples from a larger file; without it, every record in the
file is included.

## Clinical context

Clinical context is a JSON object keyed by study ID:

```json
{
  "s100": {
    "Indication": "Example indication",
    "History": "Example history",
    "Technique": "Portable AP",
    "Comparison": "Most recent prior study"
  }
}
```

Only `Indication` is consumed at training-annotation conversion and inference.
The first GPT-OSS reasoning pass can also consume `History`, `Technique`, and
`Comparison`. Missing values may be `null` or an empty string.

## Reasoning shards

Each GPT-OSS stage writes a JSON list containing stable IDs:

```json
[
  {"id": 1001, "reasoning": "Generated reasoning text."}
]
```

`merge_reasoning.py` joins shards by ID, never by list position. A correction
shard may replace an earlier value only with `--allow-overwrite`. The term
“overwrite” here concerns duplicate IDs; replacing an existing output file is
controlled independently by `--overwrite`.

## Anatomy boxes

`prepare_anatomy_boxes.py` accepts Chest ImaGenome/RGRG-style CSV, JSON, or
JSONL rows with aligned arrays:

```json
{
  "image": "files/p10/p100/s100/current.jpg",
  "bbox_coordinates": [[120, 80, 900, 940]],
  "bbox_labels": [1],
  "bbox_phrase_exists": [true],
  "bbox_is_abnormal": [false]
}
```

Labels may be 1-based indices into the 29-region list in
`prepare_anatomy_boxes.py`, or region-name strings. Pixel boxes are converted
to integer coordinates in `[0, 1000]` using the standard convention:

```text
x' = round(x / image_width * 1000)
y' = round(y / image_height * 1000)
```

The historical data-preparation script swapped width and height in this
calculation. `--legacy-swapped-dimensions` exists only to reproduce artifacts
created with that convention; new datasets should not use it.

Normalized output has this form:

```json
{
  "image": "files/p10/p100/s100/current.jpg",
  "disease": [
    {
      "bbox_labels": "right lung",
      "bbox_coordinates": [120, 80, 900, 940]
    }
  ]
}
```

## Grounded ARIVE record

The final pre-conversion record contains the current/prior image paths, report,
and grounded reasoning:

```json
{
  "id": 1001,
  "image": ["files/p10/p100/s100/current.jpg"],
  "report": "Reference findings text.",
  "reasoning": "I inspect the lungs. <|object_ref_start|>right lung<|object_ref_end|><|box_start|>120 80 900 940<|box_end|> The region is clear."
}
```

Coordinates must satisfy `0 <= x1 < x2 <= 1000` and
`0 <= y1 < y2 <= 1000`. `add_object_refs.py` first tries an exact coordinate
match and then uses label/context and coordinate similarity for legacy noisy
boxes. Review its printed `fuzzy_matched_boxes`, `unmatched_boxes`, and
`ambiguous_coordinate_matches` counts before training.

## Qwen-VL annotation

`convert_to_qwenvl.py` emits the conversation schema consumed by this repo:

```json
{
  "id": 1001,
  "image": [
    "files/p10/p100/s100/current.jpg",
    "files/p10/p100/s100/current.jpg_120_80_900_940"
  ],
  "conversations": [
    {"from": "human", "value": "...Current chest X-ray:\n<image>\n..."},
    {"from": "gpt", "value": "<think>\n...<|box_end|>\n<image>\n...\n</think>\nReference findings text."}
  ]
}
```

The suffixed crop path is virtual: there is no file named
`current.jpg_120_80_900_940`. `qwenvl/data/data_processor.py` parses the last
four underscore-separated fields, opens `current.jpg`, maps normalized
coordinates back to pixels, and creates the crop at load time. For a prior
case, the image list is ordered as prior image, current image, then all current
image crops. The number of image paths must equal the number of `<image>`
placeholders across the conversation.

## GRPO target and inference output

GRPO uses the complete assistant string from the Qwen-VL annotation as the
ground truth, including `<think>`, object references, boxes, injected `<image>`
markers, and the final answer.

For inference, `QWENVL_INFER_REPORT_PATH` selects evaluation rows and
`QWENVL_INFER_REFERENCE_REPORT_PATH` supplies the report lookup used for prior
studies. The latter may point to the full corpus. If the evaluation file is
already the full corpus, explicitly set both variables to the same path. The
repository defaults are `data/inference/evaluation_reports.jsonl` and
`data/inference/reference_reports.jsonl`, respectively.

Inference writes a JSON list:

```json
[
  {"id": 1001, "text": "<think>...</think>Generated findings."}
]
```

Set `QWENVL_INFER_SAVE_TOKEN_TRACE=1` to additionally store `token_ids` and
`action_mask`. In `action_mask`, `1` denotes a sampled action token and `0`
denotes an externally injected crop-observation token.

## MIMIC-ABN inference inputs

`inference/infer_mcot_abn.py` uses a separate evaluation JSONL whose records
must contain `question_id` and `image`. The first image is the current study;
an optional distinct second image is the prior:

```json
{
  "question_id": 13000001,
  "image": [
    "files/p10/p100/s200/current.jpg",
    "files/p10/p100/s100/prior.jpg"
  ],
  "findings": "Abnormal reference findings."
}
```

Two Prior-RRG annotations are also required. Both are JSON objects containing
`train`, `val`, and `test` lists whose records include an anchor scan:

```json
{
  "train": [],
  "val": [],
  "test": [
    {
      "findings": "Abnormal findings.",
      "anchor_scan": {
        "image_path": ["p10/p100/s200/current.jpg"]
      },
      "prior_study": null
    }
  ]
}
```

The file selected by `QWENVL_INFER_ABN_ANNOTATION_PATH` contributes only its
`test` anchor paths to define MIMIC-ABN membership. The file selected by
`QWENVL_INFER_PRIOR_ANNOTATION_PATH` is the full MIMIC-CXR annotation; anchor
and recursively nested `prior_study` image paths from every split are indexed
to obtain prior findings. Annotation paths without `files/` are normalized
before joining to evaluation paths.

The clinical-context format and image-root rules are the same as above. Run
`QWENVL_INFER_PREFLIGHT_ONLY=1 python inference/infer_mcot_abn.py` to validate
all four metadata files, the joins, unique selected `question_id` values, and
selected image files without loading the model.
