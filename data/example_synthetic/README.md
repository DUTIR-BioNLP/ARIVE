# Public synthetic one-record construction example

This example contains one internally related current study, prior study,
clinical-context object, anatomy record, few-shot record, and two tiny PGM
images. Every identifier and sentence is synthetic; no MIMIC-CXR patient data
is included.

The two expensive GPT-OSS outputs are bundled in `gpt_outputs/`. Run the
deterministic replay from the repository root:

```bash
bash scripts/run_sample_data_pipeline.sh
```

By default, the command uses this `data/example_synthetic/` fixture. It executes
coordinate normalization and Stages 2, 4, 6, 7, and 8,
using the bundled Stage 1 and Stage 5 outputs. It writes intermediates to
`data/example_synthetic/work/`, writes the final annotation to
`data/example_synthetic/output/mcot_annotation.json`, and compares it with
`expected/mcot_annotation.json`.

To validate the complete integrated command without loading GPT-OSS:

```bash
python -m data_generation.run_pipeline \
  --base-records data/example_synthetic/inputs/base_records.json \
  --reference-reports data/example_synthetic/inputs/reference_reports.json \
  --clinical-context data/example_synthetic/inputs/clinical_context.json \
  --few-shot-examples data/example_synthetic/inputs/few_shot_examples.json \
  --anatomy-source data/example_synthetic/inputs/anatomy_boxes.json \
  --image-root data/example_synthetic/inputs/images \
  --model /path/to/gpt-oss-120b \
  --work-dir data/example_synthetic/work/full-pipeline \
  --output data/example_synthetic/output/full_mcot_annotation.json \
  --dry-run
```

Remove `--dry-run` only when the GPT-OSS generation environment and model are
available. The integrated command then runs both model calls and all
deterministic stages.

## 中文说明

该目录提供一条完全合成、彼此关联的数据。无需 GPT-OSS 时，可使用仓库附带的两份
模型输出重放确定性阶段；具备 GPT-OSS 权重后，去掉 `--dry-run` 即可运行完整的两次
生成和后续转换。示例图像是 8×8 PGM 占位图，只用于验证路径和像素坐标归一化，
不具有医学含义。
