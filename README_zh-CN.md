# ARIVE： **Interleaving anatomical inspection and diagnostic reasoning for explainable radiology report generation**.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.12-3776AB?logo=python&logoColor=white" alt="Python 3.10 and 3.12">
  <img src="https://img.shields.io/badge/Base-Qwen3--VL--4B-7B61FF" alt="Qwen3-VL-4B">
  <img src="https://img.shields.io/badge/License-Apache--2.0-2F80ED" alt="Apache-2.0">
  <img src="https://img.shields.io/badge/Status-Research%20Only-B31B1B" alt="Research use only">
</p>

ARIVE 基于 Qwen3-VL-4B 生成带区域定位的胸部 X 光报告。模型输出解剖区域和框，
系统通过局部视觉分支编码对应裁剪，再注入观测继续推理。仓库包含数据构建、SFT、
GRPO 和推理代码。

[Hugging Face 模型](https://huggingface.co/akinaHCT/ARIVE) · [English](README.md) · [数据格式](docs/DATA_FORMAT.md) · [代码来源](docs/PROVENANCE.md)

> 仅限研究用途，不用于临床诊断或患者照护。

## 1. 环境配置

使用 Linux 与 NVIDIA GPU。参考环境为数据构建 Python 3.12、训练/推理 Python 3.10。
训练依赖安装如下：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/training.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

复制 `configs/env.example` 为 `.env`，填写数据和模型路径后加载；使用当前环境时，
删除或注释 `QWENVL_ENV_DIR`。所有命令均在仓库根目录执行。

```bash
set -a
source .env
set +a
```

主要代码：`data_generation/` 负责数据构建，`qwenvl/` 负责训练和 rollout，
`inference/` 负责推理，`scripts/` 与 `tools/` 提供启动脚本和工具。
现有 `mcot` 文件名和 `MCOT_*` 配置名保留，以兼容已有命令。

## 2. 推理

我们仅提供训练好的终版 [ARIVE checkpoint](https://huggingface.co/akinaHCT/ARIVE)，
包含 decoder 和 global/local vision 分支。另行获取 Qwen3-VL-4B-Instruct 基础模型，
并在本地导出 vLLM 推理目录：

```bash
export QWENVL_INFER_BASE_MODEL=/path/to/Qwen3-VL-4B-Instruct
export QWENVL_INFER_VISION_CHECKPOINT=/path/to/ARIVE
export QWENVL_INFER_VLLM_CHECKPOINT=/path/to/ARIVE-vllm

python tools/export_vllm_checkpoint.py \
  --src "$QWENVL_INFER_VISION_CHECKPOINT" \
  --dst "$QWENVL_INFER_VLLM_CHECKPOINT"
```

导出目录必须不存在或为空。保留 checkpoint 中的 tokenizer/processor 文件，以及支持
assistant 消息中 image/video 的聊天模板（`chat_template.jinja` 或 `chat_template.json`）；
导出工具会将它们复制到 vLLM 目录。

MIMIC-CXR 推理可直接从 [Libra 的 Data Download 章节](https://github.com/X-iZhang/Libra#data-download)
下载 `libra_findings_section_eval.jsonl`。若字段为 `question_id` 和 `reference`，将其
映射为 `id` 和 `findings`，保存到 `data/inference/evaluation_reports.jsonl`，或通过
`QWENVL_INFER_REPORT_PATH` 指向整理后的文件。

准备 `data/inference/{evaluation_reports.jsonl,reference_reports.jsonl,clinical_context.json}`
和 `data/images/` 图像目录，也可用 `QWENVL_INFER_*` 配置路径。Evaluation 每条包含
`id`、`image`（current 在前、可选 prior 在后）和 `findings` 或 `report`；reference
覆盖所有 prior study；clinical context 按当前 study ID 索引，仅使用 `Indication`。
样例见[数据格式](docs/DATA_FORMAT.md)。

```bash
CUDA_VISIBLE_DEVICES=0 python inference/infer_mcot.py
```

结果以 JSON list 写入 `QWENVL_INFER_OUTPUT_DIR`。默认 temperature `0.7`、top-p `0.8`、
top-k `-1`、presence penalty `1.5`；覆盖已有结果需设置 `QWENVL_INFER_OVERWRITE=1`。

MIMIC-ABN 推理需在 `.env` 中配置两个 Prior-RRG annotation 路径：ABN annotation
筛选 test 成员，完整 annotation 提供 prior findings。随后运行仅生成异常所见的入口：

```bash
export QWENVL_INFER_REPORT_PATH="$MCOT_DATA_ROOT/inference/mimic_abn_evaluation_reports.jsonl"
export QWENVL_INFER_OUTPUT_DIR="$PWD/outputs/inference_abn"
QWENVL_INFER_PREFLIGHT_ONLY=1 python inference/infer_mcot_abn.py
CUDA_VISIBLE_DEVICES=0 python inference/infer_mcot_abn.py
```

## 3. 数据构建

代码位于 `data_generation/`。默认两次 GPT-OSS 调用分别生成融合报告、临床上下文和
prior 的 reasoning，以及选择空间定位位置。`--evidence-review` 可在同一次模型加载中
增加证据修订；其余阶段完成确定性合并、归一化、校验和格式转换。

### 输入与环境

需自行准备报告、精选 few-shot 示例、临床上下文、解剖区域框及对应图像；仓库不附带
真实数据和模型权重。默认构建目录如下：

```text
data/
├── source/
│   ├── base_records.json
│   ├── few_shot_examples.json
│   ├── reference_reports.json
│   ├── clinical_context.json
│   └── anatomy_boxes.json
├── images/                       # 图像目录或软链接
├── work/pipeline/                # 中间结果
├── mcot_annotation.json          # 最终训练标注
├── example_synthetic/            # 公开合成样例
└── sample/                       # Git 忽略的本机真实样例
```

`MCOT_DATA_ROOT` 可覆盖数据根目录，单独指定的路径参数优先。完整目录约定见
[data/README.md](data/README.md)。准备目录并关联图像：

```bash
mkdir -p data/source data/inference data/work/pipeline
ln -s /path/to/MIMIC-CXR-JPG data/images
```

GPT-OSS 数据构建使用独立环境：

```bash
python -m venv .venv-generation
source .venv-generation/bin/activate
python -m pip install -r requirements/generation.txt
```

### 3.0 用一条合成或本机真实数据重放构建流程

仓库附带一组相互关联的合成 JSON、两张 8×8 PGM 占位图、两次 GPT-OSS 的预计算
输出和最终期望标注。它不包含真实患者内容，可直接验证 ID/study/path 关联、像素坐标
归一化、bbox 合并、object reference 和 Qwen-VL 转换：

```bash
bash scripts/run_sample_data_pipeline.sh
```

默认读取 `data/example_synthetic/`，结果和中间文件分别写到该目录的 `output/` 与
`work/`，并自动与 `expected/mcot_annotation.json` 比较。该命令用随仓库提供的结果
重放 Stage 1 和 Stage 5，因此不需要 GPT-OSS；完整 pipeline 与 `--dry-run` 命令见
[data/example_synthetic/README.md](data/example_synthetic/README.md)。

若本机拥有已获授权的数据，可把同一条病例的 current/prior report、clinical
context、原始 anatomy、真实 few-shot 记录及两次生成结果抽取到 Git 忽略的
`data/sample/`。脚本仅从本地图像根目录复制所选病例的 current/prior 两张图。
由于本地历史 bbox 输出使用宽高互换的归一化规则，抽取器会写入私有 marker，重放脚本
据此自动启用兼容参数：

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

该命令会在 `data/sample/work/` 和 `output/` 写入由真实病例生成的 JSON。人工确认结果
后，如需保存本机回归基准，可执行：

```bash
mkdir -p data/sample/expected
cp data/sample/output/mcot_annotation.json data/sample/expected/mcot_annotation.json
```

`data/sample/` 含受限数据，任何内容都不得提交或重新分发。

### 3.1 构造数据集

完整字段定义、prior 图像顺序、坐标规则与虚拟裁剪格式见
[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)。所有命令都应在仓库根目录以
`python -m` 形式执行。

先定义输入：

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

#### 3.1.1 一条命令运行完整 pipeline

推荐使用集成入口。它严格按照下表顺序执行默认两次 GPT-OSS 调用和六个确定性阶段，
并在每个 JSON 边界完成后立即校验：

若当前报告明显残缺（如仅有 `At the` 或拍片时间），pipeline 会在模型生成前报错，
避免把 prior 报告误当成当前发现。可传入
`--report-archive /path/to/mimic-cxr-reports.zip`，从对应 study 的原始
`IMPRESSION` 回退；若原始印象本身有歧义，可用
`--report-overrides /path/to/overrides.json` 提供按记录 ID 索引的人工核定文本。
回退记录写入工作目录的 `report_recovery_audit.json`，源文件不被修改。
bbox 合并阶段会将框的位置映射回第一阶段原文，保留全部临床描述和结论，避免第二次
模型调用改词或丢句。无法可靠对齐或不在源 anatomy 中的坐标会报错。旧工作目录需按
原代码恢复，或使用新的工作目录运行当前 pipeline。
对旧版全文预测，合并阶段还会按明确的肺区名称校正插框位置；当前句子编号方式
生成的预测保留已校验的位置。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m data_generation.run_pipeline \
  --model "$GPT_OSS_MODEL" \
  --tensor-parallel-size 8
```

该短命令直接使用 `project_paths.py` 中的 `data/` 默认值。若任一文件位于其他位置，
仍可显式传入原来的 `--base-records`、`--image-root`、`--work-dir`、`--output` 等参数。

下表是每个构造阶段真实的数据文件契约。未写环境变量前缀的中间路径均相对于
`--work-dir`，Stage 8 的目标位置则由 `--output` 决定；手工分片运行时，应把
`generated.json` 替换为所有互不重叠的分片。

| Stage | 读取的数据文件/目录 | 核心操作 | 写出的文件 | 外部计算/图像读取 |
| --- | --- | --- | --- | --- |
| 1 · `generate_reasoning` | `$BASE_RECORDS`（JSON list 或 JSONL）、`$FEW_SHOT_EXAMPLES`（JSON list 或 JSONL）、`$CLINICAL_CONTEXT`（单个 JSON object）、`$REFERENCE_REPORTS`（JSON list 或 JSONL） | 生成完整初稿，可选修订证据忠实度 | `reasoning/generated.json`、`reasoning/drafts/generated.json` | 默认一次 GPT-OSS；不读像素，图像路径只用于确定 study |
| 2 · `merge_reasoning` | `$BASE_RECORDS` 和一个或多个 Stage 1 reasoning JSON/JSONL 分片 | 按 `id` 合并 reasoning，检查重复、缺失和覆盖行为 | `reasoning.json` | 无 |
| 3 · `prepare_anatomy_boxes` | `$ANATOMY_SOURCE`（原始 CSV、JSON list 或 JSONL）及其在 `$MIMIC_CXR_ROOT` 下引用的 JPG | 读取图像宽高，把源框归一化为整数 `[0,1000]` 坐标 | `anatomy.json` | 仅打开 JPG 读取宽高 |
| 4 · `attach_anatomy_boxes` | `reasoning.json` 和 `anatomy.json`；使用 `--anatomy-is-normalized` 时，后者改为已归一化的 `$ANATOMY_SOURCE` JSON/JSONL | 按规范化后的 current-image 路径关联解剖框，并单独记录未匹配项 | `reasoning_with_boxes.json`、`missing_anatomy.json` | 无 |
| 5 · `insert_bboxes` | `reasoning_with_boxes.json` | 选择所供区域与原文定位句，再由代码插入源坐标 | `bbox/generated.json`、`bbox/raw/generated.json` | 最后一次 GPT-OSS；prompt 不读像素 |
| 6 · `build_mcot_dataset` | `reasoning_with_boxes.json` 和一个或多个 Stage 5 bbox prediction JSON/JSONL 分片 | 按 `id` 合并 bbox 预测并校验最终 ARIVE reasoning | `mcot.json` | 无 |
| 7 · `add_object_refs` | `mcot.json`，以及作为 ID/坐标/label 对照表的 `reasoning_with_boxes.json` | 为匹配到的 bbox 补上相邻的 anatomy object-reference 标签 | `mcot_with_object_refs.json` | 无 |
| 8 · `convert_to_qwenvl` | `mcot_with_object_refs.json`、`$CLINICAL_CONTEXT`、`$REFERENCE_REPORTS` | 转换为 Qwen-VL conversation，并按 prior/current/crops 顺序写入 `<image>` 和虚拟裁剪路径 | 最终 `mcot_annotation.json` | 不读像素；只写相对原图路径和虚拟裁剪路径 |

需要特别注意：这些 GPT-OSS 调用都不会读取图像像素。Stage 3 是数据构造阶段中
唯一打开 JPG 的步骤，而且只读取图像尺寸；真正的像素首次在 SFT/GRPO 或推理时
送入视觉分支。因此，记录里的图像路径都应能解析为
`$MIMIC_CXR_ROOT/<记录中的 image 路径>`。Stage 1/2/5/6 通过 `id` 关联分片；
clinical context 和 prior report 通过图像路径中的 `s<数字>` study ID 关联；
anatomy 则通过规范化后的 current image 路径关联。

运行目录还会写入 `pipeline_manifest.json`，记录输入文件身份、模型、范围和生成参数。
中断后以完全相同的参数加 `--resume` 可跳过已经存在且通过校验的阶段；不同实验应使用
新的 `--work-dir`，需要全部重算时使用 `--overwrite`。`--dry-run` 只检查输入并打印
全部阶段命令，不加载模型或写输出。若 anatomy 已经是第 3 步的归一化格式，可传
`--anatomy-is-normalized`，此时 pipeline 会跳过 `prepare_anatomy_boxes`。

#### 3.1.2 第一次 GPT：融合 few-shot、clinical context 和 prior 生成 reasoning

`BASE_RECORDS` 必须包含 `id`、当前/prior `image`、`report` 或 `findings`，不再要求
预先生成的 `reasoning`。`FEW_SHOT_EXAMPLES` 必须包含带 reasoning 的精选 JSON/JSONL
记录；如需从较大文件中选择，可重复传入 `--few-shot-example-id`。建议选两到三条
完整、有依据的逐区推理示例，展开程度应与目标输出一致，通常为 350–500 词。
精简指令不等于压缩思维链；简短的报告复述会教给模型错误的任务。示例可同时提供
`report`（或 `findings`）和 `reasoning`，展示观察如何支持解释、鉴别和不确定性，
同时避免补写未描述的正常发现。`data/example_synthetic/inputs/grounded_few_shot_examples.json`
中的短合成记录仅展示输入格式，不适合作为详细 ARIVE reasoning 的质量示例。
完整示例可通过
`--few-shot-examples data/example_synthetic/inputs/detailed_few_shot_examples.json`
使用。成对示例还可提供文本形式的 `clinical_context` 和 `prior_report`。
旧的纯 narrative 示例仍然兼容。`REFERENCE_REPORTS`
必须覆盖记录引用的所有 prior study。

生成 reasoning 时使用非空的 `Indication`、`History`、`Technique` 和 `Comparison`。
最终 Qwen-VL 用户输入只包含 `Indication`，与推理保持一致；其它临床字段仅用于
reasoning 构建。

默认提示词位于 `data_generation/prompts/merged_reasoning.txt`，要求详细展开，
不固定正文词数。自定义时，可给 `run_pipeline` 传入
`--reasoning-prompt-config /path/to/prompt.json`，或给 `generate_reasoning` 传入
`--prompt-config /path/to/prompt.json`。JSON 需包含非空的 `instruction` 和
`developer_instruction`；完整 few-shot、当前报告、临床字段和 prior 输入仍被保留。
manifest 记录提示词内容哈希，修改后不能复用旧断点。长度和格式检查不代表标注质量合格。

初稿另存于 `reasoning/drafts/`。
`--evidence-review --draft-reasoning /path/to/saved.json` 可直接修订缓存初稿，
避免重复生成。证据修订是可选的实验性文本生成，仍可能保留无依据细节，
不能代替像素级核验。
修订模型只选择需要替换的句子，其余句子和段落保留原文；原始修改存于
`reasoning/review/`。可选 `--review-notes notes.json` 接收按 ID 组织的逐例审阅意见，
这些意见是审阅输入，不是参考思维链。展开程度检查会拒绝过度压缩的修订。

```bash
python -m data_generation.generate_reasoning \
  --input "$BASE_RECORDS" \
  --clinical-context "$CLINICAL_CONTEXT" \
  --reference-reports "$REFERENCE_REPORTS" \
  --few-shot-examples "$FEW_SHOT_EXAMPLES" \
  --model "$GPT_OSS_MODEL" \
  --start 0 --end 10000 \
  --tensor-parallel-size 8 \
  --output "$PIPELINE_WORK_DIR/reasoning/000000-010000.json"

python -m data_generation.merge_reasoning \
  --base "$BASE_RECORDS" \
  --reasoning "$PIPELINE_WORK_DIR"/reasoning/*.json \
  --output "$PIPELINE_WORK_DIR/reasoning.json"
```

根据数据量重复运行互不重叠的 `[start,end)` 分片。合并脚本按稳定 ID 对齐，
不依赖列表位置。

#### 3.1.3 归一化并关联解剖框

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

默认坐标转换为 `x/image_width`、`y/image_height`。只有在复现旧版误将宽高互换
的数据时才使用 `--legacy-swapped-dimensions`。默认保留 phrase-exists 或
abnormal 标记为真的区域；`--keep-all` 会保留所有区域。

#### 3.1.4 最后一次 GPT：选择 bbox 定位句，再补齐 object reference 并转换标注

模型只选择所供区域编号和原文句子编号；代码校验编号后插入源坐标，
完整保留原有思维链，不再让模型复制整段文字。原始选择另存于 `bbox/raw/`；
通常的 bbox 分片仍包含插框后的完整文字。

```bash
python -m data_generation.insert_bboxes \
  --input "$PIPELINE_WORK_DIR/reasoning_with_boxes.json" \
  --model "$GPT_OSS_MODEL" \
  --start 0 --end 10000 \
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

python -m data_generation.convert_to_qwenvl \
  --mcot "$PIPELINE_WORK_DIR/mcot_with_object_refs.json" \
  --clinical-context "$CLINICAL_CONTEXT" \
  --reference-reports "$REFERENCE_REPORTS" \
  --output "$MCOT_DATA_ROOT/mcot_annotation.json"
```

实际执行顺序为 `insert_bboxes` → `build_mcot_dataset` → `add_object_refs`。
集成 pipeline 启用 `--preserve-source-text`，把插框位置映射回原始 reasoning；
无法可靠对齐或不在源 anatomy 中的坐标会报错。对没有
`grounding_method=source_sentence_anchors` 标记的旧预测，还会将明确命名的肺区框
重新定位到对应观察句；当前句子编号方式保留已校验的位置。
`add_object_refs` 的输入必须已有 bbox，不能提前到插框步骤之前执行。

务必检查 `add_object_refs` 输出的 fuzzy、unmatched、ambiguous 统计。标注中的
`image.jpg_x1_y1_x2_y2` 是虚拟路径，不能预先生成成图像文件；loader 会在
训练时解析后缀并从原图动态裁剪。

#### 3.1.5 一条记录贯穿完整 pipeline 的输入输出样例

下面使用合成数据展示当前代码真正读取和写出的 schema。为了突出字段变化，长篇
reasoning 仅保留与右下肺有关的代表性片段。假设 current image 是
`1000 x 800` 像素。

##### 运行前需要准备的 5 类数据

1. `BASE_RECORDS`：JSON list 或 JSONL。`image[0]` 必须是 current；有 prior 时
   `image[1]` 是 prior。没有 prior 时省略第二项，或令第二项与 current 相同。
   `report` 也可以写成 `findings`；不需要预先提供 `reasoning`。

   ```json
   [
     {
       "id": 1001,
       "image": [
         "files/p10/p100/s100/current.jpg",
         "files/p10/p100/s099/prior.jpg"
       ],
       "report": "Mild right basilar opacity, increased from the prior study."
     }
   ]
   ```

2. `FEW_SHOT_EXAMPLES`：JSON list 或 JSONL。每条示例需要 `reasoning`、`text`，
   或 `messages` 中的 assistant 文本；其中的临床事实不会作为当前病例输入。只有
   使用 `--few-shot-example-id` 筛选时，示例才必须带可匹配的 `id` 或
   `question_id`。

   ```json
   [
     {
       "id": "style-1",
       "reasoning": "We'll begin with image quality. Next I inspect the lungs, pleura, heart, and bones in order."
     }
   ]
   ```

3. `CLINICAL_CONTEXT`：必须是以 current study ID 为 key 的单个 JSON object。
   study ID 从图像路径的 `s<数字>` 目录提取。第一次 GPT 使用下列四个字段；最终
   annotation 转换只写入非空的 `Indication`。

   ```json
   {
     "s100": {
       "Indication": "Cough and fever",
       "History": "Adult patient with cough and fever",
       "Technique": "Portable AP chest radiograph",
       "Comparison": "Most recent prior study"
     }
   }
   ```

4. `REFERENCE_REPORTS`：JSON list 或 JSONL；用于通过 `image[0]` 中的 study ID
   查找 prior report。每个被 base 记录引用的 prior study 都必须存在。

   ```json
   {"id": 999, "image": ["files/p10/p100/s099/prior.jpg"], "findings": "No focal right basilar opacity."}
   ```

5. `ANATOMY_SOURCE`：CSV、JSON list 或 JSONL。坐标是原图像素坐标；box、label
   和可选 flag 数组必须按位置对齐。label 可以是解剖区域名称，也可以是代码中
   29 个区域的 1-based index。

   ```json
   [
     {
       "image": "files/p10/p100/s100/current.jpg",
       "bbox_coordinates": [[120, 160, 600, 720]],
       "bbox_labels": ["right lower lung zone"],
       "bbox_phrase_exists": [true],
       "bbox_is_abnormal": [true]
     }
   ]
   ```

这五类数据依赖三种稳定关联键：reasoning 分片通过 `id` 回连 base；clinical
context 和 prior report 通过路径中的 study ID 查找；anatomy 通过规范化后的
current image 路径匹配。

##### Stage 1：`generate_reasoning`

输入是 base 中的一条记录，以及该记录对应的 few-shot、clinical context 和可选
prior report。第一次 GPT-OSS 直接生成最终的 context/prior-aware reasoning，写入
`data/work/pipeline/reasoning/generated.json`：

```json
[
  {
    "id": 1001,
    "reasoning": "We'll begin with image quality. In the right lower lung zone, there is a mild basilar opacity. Compared with the most recent prior study, this is a new finding."
  }
]
```

##### Stage 2：`merge_reasoning`

输入是 `BASE_RECORDS` 和一个或多个 Stage 1 分片。脚本按 `id` 合并，而不是按数组
位置合并，输出 `data/work/pipeline/reasoning.json`：

```json
[
  {
    "id": 1001,
    "reasoning": "We'll begin with image quality. In the right lower lung zone, there is a mild basilar opacity. Compared with the most recent prior study, this is a new finding.",
    "report": "Mild right basilar opacity, increased from the prior study.",
    "image": [
      "files/p10/p100/s100/current.jpg",
      "files/p10/p100/s099/prior.jpg"
    ]
  }
]
```

##### Stage 3：`prepare_anatomy_boxes`

输入是 `ANATOMY_SOURCE` 和 `MIMIC_CXR_ROOT` 下的实际图像。脚本读取图像宽高，
默认按 `x/image_width`、`y/image_height` 转为整数 `[0,1000]` 坐标。上例的
`[120,160,600,720]` 因而变为 `[120,200,600,900]`，输出
`data/work/pipeline/anatomy.json`：

```json
[
  {
    "image": "files/p10/p100/s100/current.jpg",
    "disease": [
      {
        "bbox_labels": "right lower lung zone",
        "bbox_coordinates": [120, 200, 600, 900]
      }
    ]
  }
]
```

默认只保留 `bbox_phrase_exists` 或 `bbox_is_abnormal` 为真的区域；源数据没有这两组
flag 时保留所有区域。只有复现旧的宽高互换坐标时才使用
`--legacy-swapped-dimensions`。

##### Stage 4：`attach_anatomy_boxes`

输入是 Stage 2 的 reasoning 和 Stage 3 的 normalized anatomy。脚本以
`image[0]`（current）路径精确关联，将 `disease` 加入记录，输出
`data/work/pipeline/reasoning_with_boxes.json`：

```json
[
  {
    "id": 1001,
    "reasoning": "...In the right lower lung zone, there is a mild basilar opacity...",
    "report": "Mild right basilar opacity, increased from the prior study.",
    "image": [
      "files/p10/p100/s100/current.jpg",
      "files/p10/p100/s099/prior.jpg"
    ],
    "disease": [
      {
        "bbox_labels": "right lower lung zone",
        "bbox_coordinates": [120, 200, 600, 900]
      }
    ]
  }
]
```

无法匹配 anatomy 的完整 reasoning 记录会写入 `data/work/pipeline/missing_anatomy.json`；全部
匹配时该文件内容是 `[]`。集成 pipeline 默认发现非空 missing 文件即停止，只有
显式传入 `--allow-missing-anatomy` 才允许这些记录不进入后续数据。

##### Stage 5：`insert_bboxes`

输入是 Stage 4 的 `report`、`reasoning` 和 `disease`。GPT-OSS 选择区域与句子编号，
例如 `[{"region_id": 0, "sentence_id": 1}]`，原始结果写入 `bbox/raw/generated.json`。
代码据此将源坐标插到对应句子之前，输出 `data/work/pipeline/bbox/generated.json`：

```json
[
  {
    "id": 1001,
    "grounding_method": "source_sentence_anchors",
    "reasoning": "...<|box_start|>120 200 600 900<|box_end|>\nIn the right lower lung zone, there is a mild basilar opacity..."
  }
]
```

##### Stage 6：`build_mcot_dataset`

输入是 Stage 4 的完整记录和一个或多个 Stage 5 分片。脚本按 `id` 合并，规范化
bbox 标签中的空白、去除重复 box，并验证坐标范围和正面积，输出
`data/work/pipeline/mcot.json`：

```json
[
  {
    "id": 1001,
    "reasoning": "...<|box_start|>120 200 600 900<|box_end|>In the right lower lung zone, there is a mild basilar opacity...",
    "report": "Mild right basilar opacity, increased from the prior study.",
    "image": [
      "files/p10/p100/s100/current.jpg",
      "files/p10/p100/s099/prior.jpg"
    ]
  }
]
```

##### Stage 7：`add_object_refs`

输入是 Stage 6 的推理记录和 Stage 4 中保留的 bbox/label 对照。脚本先按坐标精确
匹配，旧数据坐标不一致时再使用上下文和坐标相似度匹配，在每个 box 前补上区域名，
输出 `data/work/pipeline/mcot_with_object_refs.json`：

```json
[
  {
    "id": 1001,
    "reasoning": "...<|object_ref_start|>right lower lung zone<|object_ref_end|><|box_start|>120 200 600 900<|box_end|>In the right lower lung zone, there is a mild basilar opacity...",
    "report": "Mild right basilar opacity, increased from the prior study.",
    "image": [
      "files/p10/p100/s100/current.jpg",
      "files/p10/p100/s099/prior.jpg"
    ]
  }
]
```

应检查该阶段打印的 `fuzzy_matched_boxes`、`unmatched_boxes` 和
`ambiguous_coordinate_matches`；集成 pipeline 还会强制每个 bbox 都紧邻一个
object reference。

##### Stage 8：`convert_to_qwenvl`

输入是 Stage 7 的 grounded reasoning、`CLINICAL_CONTEXT` 和
`REFERENCE_REPORTS`。脚本构造训练对话，在每个 bbox 后插入独占一行的
`<image>`，并为每个局部观测增加一个 current image 虚拟裁剪路径，得到最终
`mcot_annotation.json`：

```json
[
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
        "value": "...Indication: Cough and fever\nPrior chest X-ray:\n<image>\nPrior findings:No focal right basilar opacity.\nCurrent chest X-ray:\n<image>\n..."
      },
      {
        "from": "gpt",
        "value": "<think>\n...<|object_ref_start|>right lower lung zone<|object_ref_end|><|box_start|>120 200 600 900<|box_end|>\n<image>\nIn the right lower lung zone, there is a mild basilar opacity...\n</think>\nMild right basilar opacity, increased from the prior study."
      }
    ]
  }
]
```

有 prior 时，最终 `image` 顺序固定为 `prior, current, current crops...`；无 prior
时为 `current, current crops...`，human prompt 也不会出现 prior 段。虚拟路径不
对应磁盘上的真实文件，训练 loader 会解析末尾四个归一化坐标并从 current image
动态裁剪。整条 conversation 中 `<image>` 的数量必须与 `image` 路径数量完全一致。

以上只有 Stage 1 和 Stage 5 调用 GPT-OSS；其余阶段均为确定性合并、坐标归一化、
grounding、校验或 schema 转换。

### 3.2 训练标注

Stage 8 生成 `mcot_annotation.json`：human 输入包含 `Indication`、current/prior
图像和可用的 prior findings；assistant 包含 grounded reasoning、裁剪占位符和目标报告。
SFT/GRPO 读取这份标注及图像，不再读取 clinical context 或原始 report 文件。

GRPO 默认使用 `ABN_MIMIC_COT`，需要另行准备相同 schema 的
`abnormal_mcot_annotation.json`；也可设置 `MCOT_GRPO_DATASET_NAME=MIMIC_COT`，
直接使用 Stage 8 输出。

若已有标注由旧转换器生成，可从 `mcot_with_object_refs.json` 重新执行 Stage 8，
写入新的输出文件；已有标注不会自动修改。pipeline 版本已更新，旧 manifest 无法
通过 `--resume` 复用。

## 4. 训练

切回训练环境（`source .venv/bin/activate`）并加载 `.env`，配置 `MCOT_BASE_MODEL`、
标注、图像和 checkpoint 路径。检查 token ID 后执行 SFT 并导出：

```bash
python tools/verify_token_ids.py "$MCOT_BASE_MODEL"
CUDA_VISIBLE_DEVICES=0,1,2,3 MCOT_SFT_PREFLIGHT_ONLY=1 bash scripts/sft_qwen3_4b.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/sft_qwen3_4b.sh
python tools/export_vllm_checkpoint.py \
  --src "$MCOT_SFT_OUTPUT_DIR" --dst "$MCOT_VLLM_MODEL_PATH"
```

GRPO 需将 `MCOT_SFT_MODEL_PATH` 指向完整 SFT checkpoint，并在 `.env` 中配置
answer/reasoning CheXbert 权重和 RadGraph 资源。启动脚本要求单节点、至少四张可见 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MCOT_GRPO_PREFLIGHT_ONLY=1 bash scripts/grpo_qwen3_4b_verl.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/grpo_qwen3_4b_verl.sh
```

四项奖励为 bbox IoU、reasoning CheXbert、answer CheXbert 和 answer RadGraph，
通过 `QWENVL_GRPO_REWARD_WEIGHTS` 配置权重。SFT 自动恢复已有 `checkpoint-*`；
GRPO 需显式设置 `MCOT_GRPO_RESUME_FROM_CHECKPOINT`。用于推理时，对选定的完整
GRPO checkpoint 再运行同一导出工具，并使用配套的完整权重与 vLLM 目录。
导出目标目录必须不存在或为空。

## 检查与许可证

```bash
python -m unittest discover -s tests -v
python -m compileall -q data_generation qwenvl inference tools
bash -n scripts/sft_qwen3_4b.sh
bash -n scripts/grpo_qwen3_4b_verl.sh
bash -n scripts/run_sample_data_pipeline.sh
```

代码采用 [Apache-2.0](LICENSE)，来源说明见 [NOTICE](NOTICE)。外部模型和数据遵循
各自条款，受限数据和权重不应提交到 Git。
