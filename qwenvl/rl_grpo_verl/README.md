# GRPO rollout and rewards

This package is the bridge between the ARIVE trainer and vLLM/verl:

- `prompt_dataset.py` turns Qwen-VL conversations into prompts and held-out
  assistant targets.
- `user_rollout.py` implements iterative generation. A generated
  `<|box_end|>` pauses decoding, triggers an anchor-image crop, encodes the crop
  through the local vision branch, injects its embeddings, and resumes the
  same token trajectory.
- `rollout_backend.py` defines the backend interface and dynamic loader.
- `rewards_parallel.py` computes the four-component reward vector used by the
  released launcher: bbox IoU, reasoning CheXbert F1, answer CheXbert F1, and
  answer RadGraph F1.
- `chexbert.py` defines the answer and long-reasoning classifiers. Checkpoint
  weights are external inputs and are not distributed here.

The implementation is complete; no additional rollout code is required. Use the repository-level
`scripts/grpo_qwen3_4b_verl.sh` launcher and follow the root README for required
checkpoints, environment variables, security notes, and preflight checks.
