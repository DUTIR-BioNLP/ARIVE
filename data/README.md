# ARIVE data directory

All dataset files used by ARIVE belong under this directory. Model weights and
training checkpoints remain under `models/` or `outputs/`; they are artifacts,
not dataset inputs.

```text
data/
├── source/                         # private dataset-construction inputs
│   ├── base_records.json
│   ├── few_shot_examples.json
│   ├── reference_reports.json
│   ├── clinical_context.json
│   └── anatomy_boxes.json
├── images/                         # image tree, or a symlink to it
├── inference/                      # inference-only JSONL/JSON inputs
│   ├── evaluation_reports.jsonl
│   ├── reference_reports.jsonl
│   ├── clinical_context.json
│   ├── mimic_abn_evaluation_reports.jsonl
│   ├── priorrg_mimic_abn_annotation.json
│   └── priorrg_mimic_cxr_annotation.json
├── work/pipeline/                  # generated construction intermediates
├── mcot_annotation.json            # final SFT/full-data annotation
├── abnormal_mcot_annotation.json   # optional GRPO abnormal subset
├── example_synthetic/              # tracked public one-record fixture
└── sample/                          # ignored local real-data sample
```

`MCOT_DATA_ROOT` overrides `data/` as one unit. Relative values are resolved
from the repository root. Individual legacy environment variables and command
line flags still override the corresponding paths.

Real MIMIC-CXR files, clinical context, annotations, and generated patient-level
records are intentionally ignored by git. Do not commit restricted data.
`example_synthetic/` is the tracked, patient-free regression fixture;
`sample/` is reserved for a locally extracted real case and remains ignored.
Generate it with `tools/build_local_real_sample.py`, then replay its deterministic
stages with:

```bash
MCOT_SAMPLE_ROOT="$PWD/data/sample" bash scripts/run_sample_data_pipeline.sh
```

`inference/infer_mcot.py` uses the first three files in `inference/`.
`inference/infer_mcot_abn.py` instead uses
`mimic_abn_evaluation_reports.jsonl`, both `priorrg_*.json` annotations,
`clinical_context.json`, and the image tree. The ABN annotation selects test
membership; the full MIMIC-CXR annotation supplies prior findings.

## 中文说明

项目使用的所有数据文件统一放在 `data/` 中：构造输入放入 `source/`，图像放入
`images/`（也可以创建指向实际图像库的软链接），推理输入放入 `inference/`，
pipeline 中间结果写入 `work/pipeline/`，最终训练标注写到 `data/` 根目录。

真实 MIMIC-CXR 数据受访问协议约束，不应提交到公开仓库。`sample/` 专门保存本机
抽取的真实单样本并继续被 Git 忽略；`example_synthetic/` 才是可公开提交的合成
fixture，用于检查文件关联、坐标归一化和格式转换流程。

通用推理读取 `evaluation_reports.jsonl`、`reference_reports.jsonl` 和
`clinical_context.json`。MIMIC-ABN 推理读取 `mimic_abn_evaluation_reports.jsonl`、
两个 `priorrg_*.json`、`clinical_context.json` 和图像目录；其中 ABN annotation
负责筛选 test 成员，完整 MIMIC-CXR annotation 负责查找 prior findings。
