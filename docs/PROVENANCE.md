# Code provenance and packaging map

This file records how the experiment workspace was reorganized for this public
repository. Large generated JSON files, checkpoints, logs, virtual
environments, caches, and credentialed medical data were intentionally not
copied.

| Experiment source | Public location | Packaging changes |
| --- | --- | --- |
| `gpt_scripts/vllm_infer.py` + `gpt_scripts/clinical_info_add.py` | `data_generation/generate_reasoning.py` | Fused few-shot draft generation and clinical-context/current-prior refinement into one GPT-OSS prompt |
| `gpt_scripts/bbox_insert.py` | `data_generation/insert_bboxes.py` | Preserved the spatial prompt; generalized it from experiment-specific record selection to arbitrary shards |
| `data/dataset_source/process_cot_summary.py` | `data_generation/merge_reasoning.py` | Replaced positional list joins with validated ID-keyed joins |
| `data/dataset_source/process.py` | `data_generation/build_mcot_dataset.py` and `box_utils.py` | Separated merging, known tag repair, deduplication, and coordinate validation; correction files now use explicit precedence |
| `data/dataset_source/add_object_refs_to_mcot.py` | `data_generation/add_object_refs.py` | Retained exact/fuzzy recovery and statistics; added portable CLI and overwrite checks |
| Qwen fine-tuning `demo/dataset_construct.py` | `data_generation/convert_to_qwenvl.py` | Recovered the active ARIVE conversion from the commented experiment block and added schema validation |
| Anatomy preparation utilities | `prepare_anatomy_boxes.py` and `attach_anatomy_boxes.py` | Made width/height convention explicit and retained an opt-in legacy reproduction mode |
| `qwen-vl-finetune/scripts/sft_qwen3_4b.sh` | `scripts/sft_qwen3_4b.sh` | Removed absolute paths/background logging; added environment configuration, validation, and preflight mode |
| Qwen-VL fine-tuning modules | `qwenvl/data/` and `qwenvl/train/` | Copied the local-vision and weighted-training implementation required by the launcher; replaced the machine-specific dataset registry |
| `qwen-vl-finetune/scripts/grpo_qwen3_4b_verl.sh` | `scripts/grpo_qwen3_4b_verl.sh` | Removed machine paths; exposed topology/reward/rollout settings; added trusted-IPC warning and preflight mode |
| GRPO bridge modules | `qwenvl/rl_grpo_verl/` | Included the complete rollout, synchronization, reward, and CheXbert implementation used by the launcher |
| `scripts/mcot_vllm_external_local_mm_no_fixed_seed.py` | `inference/infer_mcot.py` | Removed checkpoint/data paths; fixed sampling-parameter plumbing; made request seeding opt-in; added input and output safety checks |
| `scripts/infer_ABN/mcot_vllm_external_local_mm_no_fixed_seed.py` | `inference/infer_mcot_abn.py` | Retained official abnormal-test selection and full-annotation prior lookup; reused the corrected single-setting rollout, strict vision loader, opt-in request seed, portable paths, and added a data-only preflight |
| Checkpoint/preflight helpers | `tools/export_vllm_checkpoint.py` and `tools/preflight_grpo.py` | Replaced hardcoded paths with guarded arguments; retained shard, processor, CUDA topology, token, and port validation |

The public scripts use the same token-level local-crop mechanism and core
training logic as the experiment code. Pipeline revisions are tracked by
`PIPELINE_VERSION` in `data_generation/run_pipeline.py`; other packaging changes
are limited to configuration, validation, recovery of the actual data
conversion, removal of machine-specific artifacts, and documented
safety/reproducibility controls. Further scientific changes such as a new
prompt, reward definition, or model architecture should be versioned separately.

The original `gpt_scripts/vllm_infer.py` and `gpt_scripts/clinical_info_add.py`
operations are represented by one fused first pass, keeping the released data
recipe at two GPT-OSS passes including bbox insertion.
