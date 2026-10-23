from __future__ import annotations

import copy
import functools
import gc
import inspect
import json
import math
import os
import random
import shutil
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import numpy as np
import torch
import torch.distributed as dist
import transformers
import torch.utils.checkpoint
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from qwenvl.rl_grpo_verl.interfaces import RolloutRequest
from qwenvl.rl_grpo_verl.prompt_dataset import GRPOPromptDataset, PromptDatasetConfig
from qwenvl.rl_grpo_verl.rewards import load_reward_batch_function, load_reward_function
from qwenvl.rl_grpo_verl.rollout_backend import load_rollout_backend
from qwenvl.data.data_processor import IMAGE_TOKEN_INDEX, load_rgb_bbox, update_processor_pixels
from qwenvl.train.argument import (
    DataArguments,
    GRPOArguments,
    ModelArguments,
    TrainingArguments,
)
from qwenvl.train.train_qwen_local import (
    attach_local_vision_branch,
    load_local_branch_weights_from_model_path,
)

try:
    from verl.trainer.ppo import core_algos
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Cannot import verl. Please install verl first, e.g. `pip install verl`."
    ) from exc

try:
    import deepspeed
except Exception:  # pragma: no cover
    deepspeed = None

try:
    from torch.utils.tensorboard import SummaryWriter as _TensorBoardSummaryWriter
except Exception:  # pragma: no cover
    _TensorBoardSummaryWriter = None

_BOX_PATTERN = re.compile(
    r"<\|box_start\|>\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*<\|box_end\|>"
)

@dataclass
class VerlBridgeArguments:
    rollout_backend_module: str = field(default="qwenvl.rl_grpo_verl.user_rollout")
    rollout_backend_class: str = field(default="UserRolloutBackend")
    reward_module: Optional[str] = field(default=None)
    reward_function_name: str = field(default="compute_reward")
    max_prompt_samples: int = field(default=-1)
    rollout_num_workers: int = field(default=0)
    save_every_steps: int = field(default=200)
    warm_start_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Load policy weights and data progress from a checkpoint while creating a fresh "
                "optimizer/ZeRO state. This supports a world-size change but is not an exact resume."
            )
        },
    )
    skip_final_save: bool = field(
        default=False,
        metadata={"help": "Skip the final model/optimizer checkpoint; intended for one-step smoke tests."},
    )


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_dist_initialized() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_dist_initialized() else 1


def _is_main_process() -> bool:
    return _get_rank() == 0


def _barrier():
    if _is_dist_initialized():
        dist.barrier()


def _print_main(*args, **kwargs):
    if _is_main_process():
        print(*args, **kwargs)


def _create_tensorboard_writer(output_dir: str):
    if not _is_main_process():
        return None
    if _TensorBoardSummaryWriter is None:
        _print_main("[warn] TensorBoard is unavailable; scalar logging disabled.")
        return None

    log_dir = os.path.join(output_dir, "tensorboard")
    try:
        writer = _TensorBoardSummaryWriter(log_dir=log_dir, flush_secs=30)
    except Exception as exc:
        _print_main(f"[warn] Failed to initialize TensorBoard writer at {log_dir}: {exc}")
        return None

    _print_main(f"[init] TensorBoard logging enabled at {log_dir}")
    return writer


def _write_tensorboard_scalars(
    writer,
    step: int,
    loss: float,
    pg: float,
    kl: float,
    avg_reward: float,
):
    if writer is None:
        return None

    try:
        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/pg", pg, step)
        writer.add_scalar("train/kl", kl, step)
        writer.add_scalar("train/avg_reward", avg_reward, step)
        writer.flush()
        return writer
    except Exception as exc:
        _print_main(f"[warn] TensorBoard logging disabled after write failure: {exc}")
        try:
            writer.close()
        except Exception:
            pass
        return None



def _compute_distributed_reward_scores(
    reward_fn: Any,
    local_reward_inputs: List[Dict[str, Any]],
    reward_batch_fn: Optional[Any] = None,
) -> List[float]:
    if _get_world_size() == 1:
        if reward_batch_fn is not None:
            return [score for score in reward_batch_fn(local_reward_inputs)]
        return [
            reward_fn(item["text"], item["ground_truth"], item["meta"])
            for item in local_reward_inputs
        ]

    gathered_reward_inputs: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(_get_world_size())]
    dist.all_gather_object(gathered_reward_inputs, local_reward_inputs)

    gathered_reward_scores: Optional[List[List[float]]] = None
    reward_error: Optional[str] = None
    if _is_main_process():
        try:
            if reward_batch_fn is not None:
                flat_inputs: List[Dict[str, Any]] = []
                split_sizes: List[int] = []
                for rank_inputs in gathered_reward_inputs:
                    rank_payload = rank_inputs or []
                    split_sizes.append(len(rank_payload))
                    flat_inputs.extend(rank_payload)

                flat_scores = reward_batch_fn(flat_inputs)
                if len(flat_scores) != len(flat_inputs):
                    raise RuntimeError(
                        "Reward batch function returned mismatched score count: "
                        f"expected {len(flat_inputs)}, got {len(flat_scores)}."
                    )

                gathered_reward_scores = []
                offset = 0
                for split_size in split_sizes:
                    next_offset = offset + split_size
                    gathered_reward_scores.append(list(flat_scores[offset:next_offset]))
                    offset = next_offset
            else:
                gathered_reward_scores = []
                for rank_inputs in gathered_reward_inputs:
                    rank_scores = []
                    for item in rank_inputs or []:
                        raw_score = reward_fn(item["text"], item["ground_truth"], item["meta"])
                        rank_scores.append(raw_score)
                    gathered_reward_scores.append(rank_scores)
        except Exception as exc:
            reward_error = (
                f"{type(exc).__name__}: {exc}\n"
                + traceback.format_exc(limit=20)
            )

    # Always broadcast an error payload as well as scores.  Without this, a
    # rank-0 reward exception leaves peer ranks blocked in the next collective.
    broadcast_payload = [{"scores": gathered_reward_scores, "error": reward_error}]
    dist.broadcast_object_list(broadcast_payload, src=0)
    payload = broadcast_payload[0]
    if payload.get("error"):
        raise RuntimeError(f"Distributed reward computation failed on rank 0:\n{payload['error']}")
    gathered_reward_scores = payload.get("scores")
    if gathered_reward_scores is None:
        raise RuntimeError("Distributed reward broadcast returned no scores.")

    local_scores = gathered_reward_scores[_get_rank()]
    if len(local_scores) != len(local_reward_inputs):
        raise RuntimeError(
            f"Distributed reward count mismatch on rank {_get_rank()}: "
            f"expected {len(local_reward_inputs)}, got {len(local_scores)}."
        )
    return [score for score in local_scores]


def _validate_reward_score_array(
    raw_scores: Any,
    expected_count: int,
    required_components: int = 4,
) -> np.ndarray:
    """Validate rewards collectively before constructing advantage tensors."""
    reward_scores: Optional[np.ndarray] = None
    local_error: Optional[str] = None
    try:
        reward_scores = np.asarray(raw_scores, dtype=np.float32)
        if reward_scores.ndim != 2:
            raise ValueError(f"expected a 2-D reward array, got shape={reward_scores.shape}")
        if int(reward_scores.shape[0]) != int(expected_count):
            raise ValueError(
                f"expected {expected_count} reward rows, got {reward_scores.shape[0]}"
            )
        if int(reward_scores.shape[1]) < int(required_components):
            raise ValueError(
                f"expected at least {required_components} reward components, "
                f"got {reward_scores.shape[1]}"
            )
        if not np.isfinite(reward_scores[:, :required_components]).all():
            bad_positions = np.argwhere(~np.isfinite(reward_scores[:, :required_components]))
            preview = bad_positions[:8].tolist()
            raise ValueError(f"reward components contain NaN/Inf at positions={preview}")
    except Exception as exc:
        local_error = f"rank={_get_rank()} {type(exc).__name__}: {exc}"

    errors: List[Optional[str]] = [local_error]
    if _get_world_size() > 1:
        errors = [None for _ in range(_get_world_size())]
        dist.all_gather_object(errors, local_error)
    failures = [error for error in errors if error]
    if failures:
        raise RuntimeError("Invalid reward tensor; refusing to update the policy. " + " | ".join(failures))
    if reward_scores is None:
        raise RuntimeError("Reward validation succeeded without producing a reward array.")
    return reward_scores


def _reward_component_weights() -> List[float]:
    """Return normalized weights for bbox/reasoning/CheXbert/RadGraph advantages."""
    raw = os.environ.get("QWENVL_GRPO_REWARD_WEIGHTS", "0.25,0.25,0.25,0.25")
    try:
        weights = [float(value.strip()) for value in raw.split(",")]
    except ValueError as exc:
        raise ValueError(
            "QWENVL_GRPO_REWARD_WEIGHTS must contain four comma-separated numbers"
        ) from exc
    if len(weights) != 4 or any(value < 0 or not math.isfinite(value) for value in weights):
        raise ValueError(
            "QWENVL_GRPO_REWARD_WEIGHTS must contain four finite non-negative values"
        )
    total = sum(weights)
    if total <= 0:
        raise ValueError("at least one GRPO reward component weight must be positive")
    return [value / total for value in weights]


def _validate_finite_tensor_collectively(name: str, tensor: torch.Tensor) -> None:
    local_error = None
    if not bool(torch.isfinite(tensor).all().item()):
        bad_count = int((~torch.isfinite(tensor)).sum().item())
        local_error = f"rank={_get_rank()} {name} contains {bad_count} NaN/Inf values"
    errors: List[Optional[str]] = [local_error]
    if _get_world_size() > 1:
        errors = [None for _ in range(_get_world_size())]
        dist.all_gather_object(errors, local_error)
    failures = [error for error in errors if error]
    if failures:
        raise RuntimeError("Invalid training tensor; refusing to update the policy. " + " | ".join(failures))


def _validate_rollout_minibatch_schedule(
    record_count: int,
    minibatch_size: int,
    inner_epochs: int,
    gradient_accumulation_steps: int,
) -> None:
    local_error = None
    if record_count <= 0:
        local_error = "rollout produced no training records"
    elif record_count % minibatch_size != 0:
        local_error = (
            f"record_count={record_count} is not divisible by minibatch_size={minibatch_size}; "
            "a short microbatch would receive the same loss weight as a full microbatch"
        )
    else:
        micro_steps = (record_count // minibatch_size) * inner_epochs
        if micro_steps % gradient_accumulation_steps != 0:
            local_error = (
                f"micro_steps={micro_steps} is not divisible by "
                f"gradient_accumulation_steps={gradient_accumulation_steps}; "
                "gradients would leak across rollout-policy versions"
            )

    errors: List[Optional[str]] = [local_error]
    if _get_world_size() > 1:
        errors = [None for _ in range(_get_world_size())]
        dist.all_gather_object(errors, local_error)
    failures = [f"rank={idx}: {error}" for idx, error in enumerate(errors) if error]
    if failures:
        raise RuntimeError("Invalid GRPO minibatch schedule. " + " | ".join(failures))


def _validate_rollout_record_counts_collectively(
    *,
    encoded_count: int,
    reward_input_count: int,
    group_index_count: int,
    completion_text_count: int,
    expected_count: int,
) -> None:
    local_counts = {
        "encoded": int(encoded_count),
        "reward_inputs": int(reward_input_count),
        "group_index": int(group_index_count),
        "completion_texts": int(completion_text_count),
    }
    local_error = None
    if len(set(local_counts.values())) != 1:
        local_error = f"local rollout record counts disagree: {local_counts}"
    elif int(encoded_count) != int(expected_count):
        local_error = (
            f"rollout produced {encoded_count} records, expected {expected_count} "
            "after completion replacement"
        )

    payloads: List[Optional[Dict[str, Any]]] = [
        {"counts": local_counts, "error": local_error}
    ]
    if _get_world_size() > 1:
        payloads = [None for _ in range(_get_world_size())]
        dist.all_gather_object(payloads, {"counts": local_counts, "error": local_error})

    failures = [
        f"rank={idx}: {payload['error']}"
        for idx, payload in enumerate(payloads)
        if payload is not None and payload.get("error")
    ]
    encoded_counts = {
        int(payload["counts"]["encoded"])
        for payload in payloads
        if payload is not None
    }
    if len(encoded_counts) != 1:
        failures.append(f"encoded record counts differ across ranks: {sorted(encoded_counts)}")
    if failures:
        raise RuntimeError("Invalid rollout record layout. " + " | ".join(failures))


def _reduce_scalar_mean(value: float) -> float:
    if _get_world_size() == 1:
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(_get_world_size())
    return float(tensor.item())


def _reduce_scalar_sum(value: float) -> float:
    if _get_world_size() == 1:
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _setup_runtime(training_args: TrainingArguments) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", getattr(training_args, "local_rank", -1)))

    if world_size > 1 and not _is_dist_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device, _get_rank(), _get_world_size()


def _load_deepspeed_config(training_args: TrainingArguments, train_micro_batch_size: int) -> Dict[str, Any]:
    ds_cfg = training_args.deepspeed
    if ds_cfg is None:
        raise ValueError("DeepSpeed config is required when initializing DeepSpeed.")

    if isinstance(ds_cfg, dict):
        config = copy.deepcopy(ds_cfg)
    else:
        with open(ds_cfg, "r") as f:
            config = json.load(f)

    grad_acc = max(1, int(training_args.gradient_accumulation_steps))
    world_size = max(1, _get_world_size())

    config["train_micro_batch_size_per_gpu"] = int(train_micro_batch_size)
    config["gradient_accumulation_steps"] = grad_acc
    config["train_batch_size"] = int(train_micro_batch_size) * grad_acc * world_size
    if "gradient_clipping" not in config and getattr(training_args, "max_grad_norm", None) is not None:
        config["gradient_clipping"] = float(training_args.max_grad_norm)
    return config


def _init_deepspeed_engine(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    training_args: TrainingArguments,
    train_micro_batch_size: int,
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Dict[str, Any]]:
    if deepspeed is None:
        raise ImportError("Cannot import deepspeed. Please install deepspeed first.")

    ds_config = _load_deepspeed_config(training_args, train_micro_batch_size=train_micro_batch_size)
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    if optimizer is None:
        model_parameters = list(model.parameters())
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        optimizer=optimizer,
        config=ds_config,
    )
    return engine, optimizer, ds_config


def _apply_tuning_flags(model: torch.nn.Module, model_args: ModelArguments):
    # Keep the same trainability behavior as existing SFT entry.
    if hasattr(model, "visual"):
        for _, p in model.visual.named_parameters():
            p.requires_grad = bool(model_args.tune_mm_vision)
        if hasattr(model.visual, "merger"):
            for _, p in model.visual.merger.named_parameters():
                p.requires_grad = bool(model_args.tune_mm_mlp)

    if hasattr(model, "language_model"):
        for _, p in model.language_model.named_parameters():
            p.requires_grad = bool(model_args.tune_mm_llm)
    if hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = bool(model_args.tune_mm_llm)

    if hasattr(model, "model") and hasattr(model.model, "local_visual"):
        tune_local_vision = (
            model_args.tune_mm_vision
            if model_args.tune_mm_local_vision is None
            else model_args.tune_mm_local_vision
        )
        tune_local_mlp = (
            model_args.tune_mm_mlp
            if model_args.tune_mm_local_mlp is None
            else model_args.tune_mm_local_mlp
        )
        for _, p in model.model.local_visual.named_parameters():
            p.requires_grad = bool(tune_local_vision)
        if hasattr(model.model.local_visual, "merger"):
            for _, p in model.model.local_visual.merger.named_parameters():
                p.requires_grad = bool(tune_local_mlp)

    if not (
        model_args.tune_mm_llm
        or model_args.tune_mm_mlp
        or model_args.tune_mm_vision
        or bool(model_args.tune_mm_local_vision)
        or bool(model_args.tune_mm_local_mlp)
    ):
        # If user passes no tune flags, default to full finetune.
        for p in model.parameters():
            p.requires_grad = True


def _is_peft_model(model: torch.nn.Module) -> bool:
    return hasattr(model, "peft_config") and hasattr(model, "save_pretrained")


def _collect_named_linear_modules(model: torch.nn.Module, module_key: str) -> List[str]:
    targets: List[str] = []
    for module_name, module in model.named_modules():
        if module_key in module_name and isinstance(module, torch.nn.Linear):
            targets.append(module_name)
    return targets


def _collect_lora_target_modules(model: torch.nn.Module, use_local_vision_branch: bool) -> List[str]:
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    visual_lora_targets = _collect_named_linear_modules(model, "visual")
    if visual_lora_targets:
        lora_targets.extend(visual_lora_targets)
        _print_main(f"Added {len(visual_lora_targets)} visual LoRA target modules")
    if use_local_vision_branch:
        local_visual_lora_targets = _collect_named_linear_modules(model, "local_visual")
        if local_visual_lora_targets:
            lora_targets.extend(local_visual_lora_targets)
            _print_main(f"Added {len(local_visual_lora_targets)} local_visual LoRA target modules")
    return lora_targets


def _enable_lora(
    model: torch.nn.Module,
    model_args: ModelArguments,
    training_args: TrainingArguments,
) -> torch.nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:  # pragma: no cover
        raise ImportError("LoRA requested but `peft` is not installed. Please `pip install peft`.") from exc

    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=training_args.lora_r or 64,
        lora_alpha=training_args.lora_alpha or 128,
        lora_dropout=training_args.lora_dropout or 0.05,
        target_modules=_collect_lora_target_modules(model, model_args.use_local_vision_branch),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    _print_main("LoRA enabled")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def _load_policy_model(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    local_branch_model_path: Optional[str] = None,
    local_branch_strict_load: Optional[bool] = None,
) -> tuple[torch.nn.Module, Any]:
    model_name = model_args.model_name_or_path
    dtype = torch.bfloat16 if training_args.bf16 else None
    use_local_branch = bool(data_args.enable_local_image or model_args.use_local_vision_branch)

    if "qwen3" in model_name.lower() and "a" in Path(model_name.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=dtype,
        )
        if use_local_branch:
            model = attach_local_vision_branch(model)
    elif "qwen3" in model_name.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=dtype,
        )
        if use_local_branch:
            model = attach_local_vision_branch(model)
    elif "qwen2.5" in model_name.lower():
        if use_local_branch:
            raise NotImplementedError("Local vision branch is currently only wired for Qwen3-VL in this GRPO entry.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=dtype,
        )
    else:
        if use_local_branch:
            raise NotImplementedError("Local vision branch is currently only wired for Qwen3-VL in this GRPO entry.")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            dtype=dtype,
        )

    if use_local_branch:
        if not hasattr(model, "model") or not hasattr(model.model, "local_visual"):
            raise ValueError("Local vision branch was requested but failed to attach.")
        local_branch_path = local_branch_model_path or model_args.local_branch_model_path
        strict_local_branch = (
            bool(model_args.local_branch_strict_load)
            if local_branch_strict_load is None
            else bool(local_branch_strict_load)
        )
        load_local_branch_weights_from_model_path(
            model,
            local_branch_path,
            strict_on_no_match=strict_local_branch,
        )

    processor = AutoProcessor.from_pretrained(model_name, fix_mistral_regex=True)
    data_args.min_pixels = 784
    data_args.max_pixels = 50176
    processor = update_processor_pixels(processor, data_args)
    _apply_tuning_flags(model, model_args)
    model.config.use_cache = False
    return model, processor


def _enable_gradient_checkpointing(
    model: torch.nn.Module,
    training_args: TrainingArguments,
):
    if not getattr(training_args, "gradient_checkpointing", False):
        return

    gradient_checkpointing_kwargs = getattr(training_args, "gradient_checkpointing_kwargs", None)
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}
    gradient_checkpointing_func = functools.partial(
        torch.utils.checkpoint.checkpoint,
        **gradient_checkpointing_kwargs,
    )

    def _set_gc_state(module: Optional[torch.nn.Module], enable: bool) -> bool:
        if module is None:
            return False
        is_set = False
        if hasattr(module, "gradient_checkpointing"):
            module._gradient_checkpointing_func = gradient_checkpointing_func
            module.gradient_checkpointing = enable
            is_set = True
        for submodule in module.modules():
            if hasattr(submodule, "gradient_checkpointing"):
                submodule._gradient_checkpointing_func = gradient_checkpointing_func
                submodule.gradient_checkpointing = enable
                is_set = True
        return is_set

    language_model = getattr(model, "language_model", None)
    visual_model = getattr(model, "visual", None)
    llm_gc_enabled = _set_gc_state(language_model, True)
    visual_gc_disabled = _set_gc_state(visual_model, False)

    if llm_gc_enabled:
        # Keep top-level flags disabled so only the language tower participates.
        if hasattr(model, "gradient_checkpointing"):
            model._gradient_checkpointing_func = gradient_checkpointing_func
            model.gradient_checkpointing = False
        base_model = getattr(model, "model", None)
        if base_model is not None and hasattr(base_model, "gradient_checkpointing"):
            base_model._gradient_checkpointing_func = gradient_checkpointing_func
            base_model.gradient_checkpointing = False
        _print_main(
            f"[init] gradient checkpointing enabled for language_model only "
            f"(visual_disabled={int(visual_gc_disabled)})"
        )
    elif hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        _print_main("[init] gradient checkpointing enabled via model default path")

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, inputs, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def _unwrap_forward_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _supports_model_forward_kwarg(model: torch.nn.Module, kwarg_name: str) -> bool:
    target = _unwrap_forward_model(model)
    forward = getattr(target, "forward", None)
    if forward is None:
        return False
    try:
        return kwarg_name in inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False


def _get_bbox(text: str):
    """Split model text into bbox list, cleaned think text, and final answer."""
    content = text or ""
    bbox_list = [[int(coord) for coord in match.groups()] for match in _BOX_PATTERN.finditer(content)]

    return bbox_list

def _load_rollout_bbox_image(anchor_image: str, bbox=None):
    """Return the shared unresized RGB crop used by SFT and rollout."""
    with Image.open(anchor_image) as image:
        rgb_image = image.convert("RGB")
    return load_rgb_bbox(rgb_image, bbox)

def _build_full_messages_with_assistant_content(
    messages: List[Dict[str, Any]],
    content: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    full_messages = copy.deepcopy(messages)
    full_messages.append({"role": "assistant", "content": content})
    return full_messages


def _assistant_content_to_text(content: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", "")
            if text:
                parts.append(str(text))
        elif item_type == "image":
            parts.append("<image>")
        elif item_type == "video":
            parts.append("<video>")
    # Text fragments already contain the exact separators around multimodal
    # observations.  Adding another newline here changes the rewarded rollout.
    return "".join(parts)


def _truncate_text_item_to_fit(
    processor: Any,
    messages: List[Dict[str, Any]],
    prefix_content: List[Dict[str, Any]],
    text: str,
    max_total_length: int,
) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return ""

    best_text = ""
    lo, hi = 0, len(token_ids)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate_text = tokenizer.decode(
            token_ids[:mid],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()
        candidate_content = list(prefix_content)
        if candidate_text:
            candidate_content.append({"type": "text", "text": candidate_text})
        candidate_messages = _build_full_messages_with_assistant_content(messages, candidate_content)
        candidate_res = processor.apply_chat_template(
            candidate_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        candidate_len = int(candidate_res["input_ids"].shape[1])
        if candidate_len <= max_total_length:
            best_text = candidate_text
            lo = mid + 1
        else:
            hi = mid - 1
    return best_text


def _truncate_assistant_content_to_budget(
    messages: List[Dict[str, Any]],
    content: List[Dict[str, Any]],
    processor: Any,
    max_total_length: int,
) -> List[Dict[str, Any]]:
    truncated_content = copy.deepcopy(content)
    while truncated_content:
        candidate_messages = _build_full_messages_with_assistant_content(messages, truncated_content)
        candidate_res = processor.apply_chat_template(
            candidate_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        candidate_len = int(candidate_res["input_ids"].shape[1])
        if candidate_len <= max_total_length:
            return truncated_content

        last_item = truncated_content[-1]
        if not isinstance(last_item, dict):
            truncated_content.pop()
            continue
        if last_item.get("type") != "text":
            truncated_content.pop()
            continue

        prefix_content = truncated_content[:-1]
        fitted_text = _truncate_text_item_to_fit(
            processor=processor,
            messages=messages,
            prefix_content=prefix_content,
            text=last_item.get("text", ""),
            max_total_length=max_total_length,
        )
        if fitted_text:
            truncated_content[-1] = {"type": "text", "text": fitted_text}
            return truncated_content
        truncated_content.pop()
    return truncated_content


def _append_assistant_message(
    messages: List[Dict[str, Any]],
    completion_text: str,
    anchor_image: str,
    processor: Optional[Any] = None,
    max_total_length: Optional[int] = None,
) -> tuple[List[Dict[str, Any]], List[int]]:
    bbox_list = _get_bbox(completion_text)
    image_placeholder_count = completion_text.count("<image>")
    if image_placeholder_count > len(bbox_list):
        missing_image_count = image_placeholder_count - len(bbox_list)
        bbox_list = list(bbox_list) + [None] * missing_image_count
        print("[Warn] Number of <image> placeholders exceeds the number of provided images")

    image_pool = []
    image_branch_pool: List[int] = []
    for bbox in bbox_list:
        image_pool.append({
            "type": "image",
            "image": _load_rollout_bbox_image(anchor_image, bbox),
        })
        # A missing bbox means the full anchor image was loaded and must use
        # the global visual branch.  Parsed crops use the local branch.
        image_branch_pool.append(0 if bbox is None else 1)

    video_pool = []


    content = []
    completion_image_branch_flags: List[int] = []
    # Split text by <image> or <video> placeholders while keeping delimiters
    text_parts = re.split(r"(<image>|<video>)", completion_text)
    for seg in text_parts:
        if seg == "<image>":
            if not image_pool:
                raise ValueError(
                    "Number of <image> placeholders exceeds the number of provided images"
                )
            content.append(image_pool.pop(0))
            completion_image_branch_flags.append(image_branch_pool.pop(0))
        elif seg == "<video>":
            if not video_pool:
                raise ValueError(
                    "Number of <video> placeholders exceeds the number of provided videos"
                )
            content.append(video_pool.pop(0))
        elif seg:
            # Preserve the rollout text byte-for-byte.  Stripping here can
            # change tokenization and invalidates old-policy log-probabilities.
            content.append({"type": "text", "text": seg})
    if processor is not None and max_total_length is not None and max_total_length > 0:
        content = _truncate_assistant_content_to_budget(
            messages=messages,
            content=content,
            processor=processor,
            max_total_length=int(max_total_length),
        )
        # Truncation removes only a suffix of content, so retained image flags
        # are the corresponding prefix.
        retained_image_count = sum(
            isinstance(item, dict) and item.get("type") == "image"
            for item in content
        )
        completion_image_branch_flags = completion_image_branch_flags[:retained_image_count]

    return (
        _build_full_messages_with_assistant_content(messages, content),
        completion_image_branch_flags,
    )


def _count_image_placeholders(value: Any) -> int:
    if isinstance(value, str):
        return value.count("<image>")
    if isinstance(value, dict):
        return sum(_count_image_placeholders(v) for v in value.values())
    if isinstance(value, list):
        return sum(_count_image_placeholders(v) for v in value)
    return 0


ANSWER_START_TOKEN = 77091
ANSWER_END_TOKEN = 151645
END_OF_TEXT_TOKEN = 151643
THINK_START_TOKEN = 151667
THINK_END_TOKEN = 151668
DEEPSPEED_CHECKPOINT_TAG = "ds_state"


class InvalidRolloutTraceError(RuntimeError):
    """A completion cannot be aligned safely with its exact rollout trace."""


class InvalidCompatibilityCompletionError(RuntimeError):
    """A text-only compatibility completion cannot be encoded safely."""


def _is_out_of_memory_error(exc: BaseException) -> bool:
    """Keep allocator failures fatal instead of treating them as bad completions."""
    if isinstance(exc, MemoryError):
        return True

    oom_types = []
    for owner in (torch, getattr(torch, "cuda", None)):
        oom_type = getattr(owner, "OutOfMemoryError", None)
        if isinstance(oom_type, type):
            oom_types.append(oom_type)
    if oom_types and isinstance(exc, tuple(oom_types)):
        return True

    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 12:
        return True

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cannot allocate memory",
            "can't allocate memory",
            "defaultcpuallocator: not enough memory",
        )
    )


def _find_assistant_content_bounds(full_ids: torch.Tensor, prompt_len: int) -> tuple[int, int]:
    """Return [content_start, template_eos) for the newly appended assistant."""
    seq_len = int(full_ids.numel())
    for pos in range(max(int(prompt_len), 0), seq_len):
        if int(full_ids[pos]) != ANSWER_START_TOKEN:
            continue
        content_start = pos + 2
        if content_start > seq_len:
            break
        content_end = content_start
        while content_end < seq_len and int(full_ids[content_end]) != ANSWER_END_TOKEN:
            content_end += 1
        if content_end < seq_len:
            return content_start, content_end
    raise InvalidRolloutTraceError(
        "Could not locate the appended assistant span in processed input_ids."
    )


def _image_pad_run_lengths(full_ids: torch.Tensor, start: int, end: int) -> List[int]:
    """Collect processor-expanded image-pad run lengths in one assistant span."""
    run_lengths: List[int] = []
    pos = int(start)
    while pos < int(end):
        if int(full_ids[pos]) != IMAGE_TOKEN_INDEX:
            pos += 1
            continue
        run_end = pos + 1
        while run_end < int(end) and int(full_ids[run_end]) == IMAGE_TOKEN_INDEX:
            run_end += 1
        run_lengths.append(run_end - pos)
        pos = run_end
    return run_lengths


def _build_compatibility_completion_mask(
    full_ids: torch.Tensor,
    prompt_len: int,
    tokenizer: Any,
) -> torch.Tensor:
    """Mask text-only backend output with the same action semantics as SFT.

    Assistant text and its template EOS are actions.  Dynamically inserted
    visual spans, including the single protocol newline immediately before
    and after each span, are environment observations and remain masked.
    """
    try:
        content_start, template_eos = _find_assistant_content_bounds(full_ids, prompt_len)
    except InvalidRolloutTraceError as exc:
        raise InvalidCompatibilityCompletionError(str(exc)) from exc

    completion_mask = torch.zeros(full_ids.numel(), dtype=torch.float32)
    completion_mask[content_start : template_eos + 1] = 1.0

    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    ids = full_ids.tolist()
    cursor = content_start
    while cursor < template_eos:
        if int(ids[cursor]) != int(vision_start_id):
            cursor += 1
            continue

        observation_end = cursor + 1
        while observation_end < template_eos and int(ids[observation_end]) != int(vision_end_id):
            observation_end += 1
        if observation_end >= template_eos:
            completion_mask[cursor:template_eos] = 0.0
            break

        mask_start = cursor
        newline_start = cursor - len(newline_ids)
        if (
            newline_ids
            and newline_start >= content_start
            and ids[newline_start:cursor] == newline_ids
        ):
            mask_start = newline_start

        mask_end = observation_end + 1
        if newline_ids and ids[mask_end : mask_end + len(newline_ids)] == newline_ids:
            mask_end += len(newline_ids)
        completion_mask[mask_start:mask_end] = 0.0
        cursor = mask_end

    return completion_mask


def _rebuild_with_rollout_token_trace(
    full_ids: torch.Tensor,
    prompt_len: int,
    rollout_token_ids: List[int],
    rollout_action_mask: Optional[List[int]],
    max_completion_length: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace re-tokenized assistant text with the exact rollout trajectory.

    The rollout trace uses one logical ``<|image_pad|>`` per injected image,
    while the HF processor expands it to the feature-token count required by
    each image.  Expand only observation pads (mask=0), then splice the exact
    sampled IDs into the processed multimodal sequence.
    """
    try:
        trace_ids = [int(token_id) for token_id in rollout_token_ids]
        if rollout_action_mask is None:
            trace_mask = [1] * len(trace_ids)
        else:
            trace_mask = [int(value) for value in rollout_action_mask]
    except (TypeError, ValueError) as exc:
        raise InvalidRolloutTraceError(
            "Rollout token_ids/action_mask must contain integer-compatible values."
        ) from exc
    if len(trace_ids) != len(trace_mask):
        raise InvalidRolloutTraceError(
            "Rollout token/action-mask length mismatch: "
            f"token_ids={len(trace_ids)}, action_mask={len(trace_mask)}."
        )
    if any(value not in (0, 1) for value in trace_mask):
        raise InvalidRolloutTraceError("Rollout action_mask must contain only 0/1 values.")
    action_count = sum(trace_mask)
    if max_completion_length and max_completion_length > 0 and action_count > int(max_completion_length):
        raise InvalidRolloutTraceError(
            "Rollout action trace exceeds max_completion_length: "
            f"actions={action_count}, limit={int(max_completion_length)}."
        )

    content_start, template_eos = _find_assistant_content_bounds(full_ids, prompt_len)
    image_pad_runs = _image_pad_run_lengths(full_ids, content_start, template_eos)
    logical_observation_pads = sum(
        token_id == IMAGE_TOKEN_INDEX and mask_value == 0
        for token_id, mask_value in zip(trace_ids, trace_mask)
    )
    if logical_observation_pads != len(image_pad_runs):
        raise InvalidRolloutTraceError(
            "Rollout observation/image mismatch: "
            f"trace_images={logical_observation_pads}, processed_images={len(image_pad_runs)}."
        )

    expanded_ids: List[int] = []
    expanded_mask: List[int] = []
    image_idx = 0
    for token_id, mask_value in zip(trace_ids, trace_mask):
        if token_id == IMAGE_TOKEN_INDEX and mask_value == 0:
            repeat = image_pad_runs[image_idx]
            image_idx += 1
            expanded_ids.extend([token_id] * repeat)
            expanded_mask.extend([0] * repeat)
        else:
            expanded_ids.append(token_id)
            expanded_mask.append(mask_value)

    # vLLM includes a sampled stop token in token_ids but excludes it from
    # decoded text.  If EOS was sampled, replace the chat template EOS with
    # that action; otherwise retain the template EOS as a masked token.
    sampled_eos = bool(
        expanded_ids
        and expanded_ids[-1] in (ANSWER_END_TOKEN, END_OF_TEXT_TOKEN)
        and expanded_mask[-1] == 1
    )
    suffix_start = template_eos + 1 if sampled_eos else template_eos
    rebuilt_ids = (
        full_ids[:content_start].tolist()
        + expanded_ids
        + full_ids[suffix_start:].tolist()
    )
    completion_mask = (
        [0] * content_start
        + expanded_mask
        + [0] * (int(full_ids.numel()) - suffix_start)
    )
    if len(rebuilt_ids) != len(completion_mask):
        raise InvalidRolloutTraceError(
            "Internal rollout trace splice produced misaligned IDs and action mask."
        )
    return (
        torch.tensor(rebuilt_ids, dtype=full_ids.dtype),
        torch.tensor(completion_mask, dtype=torch.float32),
    )


def _find_subsequence(sequence: List[int], pattern: List[int], start: int = 0) -> int:
    if not pattern:
        return -1
    max_start = len(sequence) - len(pattern)
    for idx in range(max(start, 0), max_start + 1):
        if sequence[idx : idx + len(pattern)] == pattern:
            return idx
    return -1


def _build_think_segment_masks(
    full_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    think_mask = torch.zeros_like(completion_mask)
    answer_mask = torch.zeros_like(completion_mask)

    completion_positions = completion_mask.nonzero(as_tuple=False).flatten()
    if completion_positions.numel() == 0:
        return think_mask, answer_mask

    completion_token_ids = full_ids[completion_positions].tolist()
    think_open_ids = [THINK_START_TOKEN]
    think_close_ids = [THINK_END_TOKEN]
    think_start = _find_subsequence(completion_token_ids, think_open_ids)
    if think_start < 0:
        answer_mask.copy_(completion_mask)
        return think_mask, answer_mask

    think_end = _find_subsequence(
        completion_token_ids,
        think_close_ids,
        start=think_start + len(think_open_ids),
    )
    if think_end < 0:
        answer_mask.copy_(completion_mask)
        return think_mask, answer_mask

    think_end_inclusive = think_end + len(think_close_ids) - 1
    think_positions = completion_positions[think_start : think_end_inclusive + 1]
    answer_positions = completion_positions[think_end + len(think_close_ids) :]
    think_mask[think_positions] = 1.0
    answer_mask[answer_positions] = 1.0
    return think_mask, answer_mask


def _count_processed_mm_items(processed: Dict[str, Any], grid_key: str) -> int:
    grid = processed.get(grid_key)
    if grid is None:
        return 0
    if not isinstance(grid, torch.Tensor):
        try:
            return len(grid)
        except Exception:
            return 0
    if grid.ndim == 0:
        return 0
    if grid.ndim == 1:
        return int(grid.numel() // 3) if grid.numel() % 3 == 0 else int(grid.numel())
    if grid.ndim >= 2:
        return int(grid.reshape(-1, grid.shape[-1]).shape[0])
    return 0


def _validate_processed_image_tensors(
    processed: Dict[str, Any],
    image_branch_flags: List[int],
    error_type: type[RuntimeError],
) -> None:
    pixel_values = processed.get("pixel_values")
    image_grid_thw = processed.get("image_grid_thw")
    if (pixel_values is None) != (image_grid_thw is None):
        raise error_type("pixel_values and image_grid_thw must either both be present or both be absent.")
    if image_grid_thw is None:
        if image_branch_flags:
            raise error_type(
                f"image_branch_flags were produced without image tensors: flags={len(image_branch_flags)}."
            )
        return
    if not isinstance(pixel_values, torch.Tensor) or not isinstance(image_grid_thw, torch.Tensor):
        raise error_type("Processed pixel_values and image_grid_thw must be tensors.")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[-1] != 3:
        raise error_type(
            f"image_grid_thw must have shape [num_images, 3], got {tuple(image_grid_thw.shape)}."
        )
    image_count = int(image_grid_thw.shape[0])
    if len(image_branch_flags) != image_count:
        raise error_type(
            "image_branch_flags/image_grid_thw count mismatch: "
            f"flags={len(image_branch_flags)}, images={image_count}."
        )
    if any(int(flag) not in (0, 1) for flag in image_branch_flags):
        raise error_type("image_branch_flags must contain only 0 (global) or 1 (local).")
    expected_patch_rows = int(image_grid_thw.prod(-1).sum().item())
    actual_patch_rows = int(pixel_values.shape[0]) if pixel_values.ndim >= 1 else 0
    if actual_patch_rows != expected_patch_rows:
        raise error_type(
            "pixel_values/image_grid_thw patch-row mismatch: "
            f"pixel_rows={actual_patch_rows}, grid_product_sum={expected_patch_rows}."
        )


def _encode_sample(
    processor: Any,
    prompt_messages: List[Dict[str, Any]],
    completion_text: str,
    anchor_image: str,
    completion_token_ids: Optional[List[int]] = None,
    completion_action_mask: Optional[List[int]] = None,
    image_branch_flags: Optional[List[int]] = None,
    max_completion_length: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    prompt_res = processor.apply_chat_template(
        prompt_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_len = int(prompt_res["input_ids"].shape[1])

    compatibility_path = completion_token_ids is None
    completion_image_branch_flags: List[int] = []
    try:
        full_messages, completion_image_branch_flags = _append_assistant_message(
            prompt_messages,
            completion_text,
            anchor_image,
            processor=processor,
            max_total_length=(
                prompt_len + int(max_completion_length)
                if (
                    compatibility_path
                    and max_completion_length
                    and max_completion_length > 0
                )
                else None
            ),
        )

        full_res = processor.apply_chat_template(
            full_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_ids = full_res["input_ids"][0]
    except (ValueError, TypeError, IndexError, OSError, RuntimeError) as exc:
        if _is_out_of_memory_error(exc):
            raise
        error_type = (
            InvalidCompatibilityCompletionError
            if compatibility_path
            else InvalidRolloutTraceError
        )
        raise error_type(
            f"Completion multimodal encoding failed with {type(exc).__name__}: {exc}"
        ) from exc

    if full_ids.numel() <= prompt_len:
        return None

    effective_completion_text = ""
    if full_messages and isinstance(full_messages[-1], dict):
        assistant_content = full_messages[-1].get("content", [])
        if isinstance(assistant_content, list):
            effective_completion_text = _assistant_content_to_text(assistant_content)

    if completion_token_ids is not None:
        full_ids, completion_mask = _rebuild_with_rollout_token_trace(
            full_ids=full_ids,
            prompt_len=prompt_len,
            rollout_token_ids=completion_token_ids,
            rollout_action_mask=completion_action_mask,
            max_completion_length=max_completion_length,
        )
    else:
        # Compatibility path for rollout backends that do not expose token
        # traces. Interactive GRPO uses the exact path above.
        completion_mask = _build_compatibility_completion_mask(
            full_ids=full_ids,
            prompt_len=prompt_len,
            tokenizer=getattr(processor, "tokenizer", processor),
        )

    think_completion_mask, answer_completion_mask = _build_think_segment_masks(
        full_ids=full_ids,
        completion_mask=completion_mask,
    )
    if float(completion_mask.sum().item()) <= 0:
        return None

    prompt_image_count = _count_processed_mm_items(prompt_res, "image_grid_thw")
    if prompt_image_count <= 0:
        prompt_image_count = _count_image_placeholders(prompt_messages)

    full_image_count = _count_processed_mm_items(full_res, "image_grid_thw")
    error_type = (
        InvalidCompatibilityCompletionError
        if compatibility_path
        else InvalidRolloutTraceError
    )
    if full_image_count < prompt_image_count:
        raise error_type(
            "Processed full sample contains fewer images than its prompt: "
            f"full={full_image_count}, prompt={prompt_image_count}."
        )
    completion_image_count = full_image_count - prompt_image_count
    if len(completion_image_branch_flags) != completion_image_count:
        raise error_type(
            "Completion image/branch-flag count mismatch: "
            f"flags={len(completion_image_branch_flags)}, images={completion_image_count}."
        )

    if image_branch_flags is None:
        prompt_flags = [0] * prompt_image_count
    else:
        prompt_flags = [int(flag) for flag in image_branch_flags]
        if len(prompt_flags) != prompt_image_count:
            raise error_type(
                "Prompt image/branch-flag count mismatch: "
                f"flags={len(prompt_flags)}, images={prompt_image_count}."
            )
    image_branch_flags = prompt_flags + completion_image_branch_flags
    _validate_processed_image_tensors(
        processed=full_res,
        image_branch_flags=image_branch_flags,
        error_type=error_type,
    )

    sample = {
        "input_ids": full_ids,
        "completion_mask": completion_mask,
        "think_completion_mask": think_completion_mask,
        "answer_completion_mask": answer_completion_mask,
        "effective_completion_text": effective_completion_text,
    }

    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if key in full_res and full_res[key] is not None:
            sample[key] = full_res[key]

    if image_branch_flags:
        sample["image_branch_flags"] = torch.tensor(image_branch_flags, dtype=torch.long)
    return sample


def _collate_mm_samples(samples: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    input_ids = [s["input_ids"] for s in samples]
    attention_masks = [torch.ones(ids.numel(), dtype=torch.long) for ids in input_ids]
    completion_masks = [s["completion_mask"] for s in samples]
    think_completion_masks = [s["think_completion_mask"] for s in samples]
    answer_completion_masks = [s["answer_completion_mask"] for s in samples]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0.0)
    completion_masks = pad_sequence(completion_masks, batch_first=True, padding_value=0.0)
    think_completion_masks = pad_sequence(think_completion_masks, batch_first=True, padding_value=0.0)
    answer_completion_masks = pad_sequence(answer_completion_masks, batch_first=True, padding_value=0.0)
    batch: Dict[str, Any] = {
        "input_ids": input_ids,
        # Qwen uses <|endoftext|> (151643) as pad_token_id and also as a
        # legitimate secondary EOS/action.  Infer padding from sequence
        # lengths, never from token value.
        "attention_mask": attention_masks,
        "completion_mask": completion_masks,
        "think_completion_mask": think_completion_masks,
        "answer_completion_mask": answer_completion_masks,
    }

    images = [s["pixel_values"] for s in samples if "pixel_values" in s]
    if images:
        batch["pixel_values"] = torch.cat(images, dim=0)
        batch["image_grid_thw"] = torch.cat([s["image_grid_thw"] for s in samples if "image_grid_thw" in s], dim=0)

    videos = [s["pixel_values_videos"] for s in samples if "pixel_values_videos" in s]
    if videos:
        batch["pixel_values_videos"] = torch.cat(videos, dim=0)
        batch["video_grid_thw"] = torch.cat([s["video_grid_thw"] for s in samples if "video_grid_thw" in s], dim=0)

    flags = [s["image_branch_flags"] for s in samples if "image_branch_flags" in s]
    if flags:
        batch["image_branch_flags"] = torch.cat(flags, dim=0)

    return batch


def _collate_train_records(records: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    batch = _collate_mm_samples(records, pad_token_id=pad_token_id)

    # old_log_probs = pad_sequence(
    #     [r["old_log_probs"] for r in records], batch_first=True, padding_value=0.0
    # )
    advantages = pad_sequence([r["advantages"] for r in records], batch_first=True, padding_value=0.0)
    response_mask = pad_sequence([r["response_mask"] for r in records], batch_first=True, padding_value=0.0)
    think_response_mask = pad_sequence(
        [r["think_response_mask"] for r in records], batch_first=True, padding_value=0.0
    )
    answer_response_mask = pad_sequence(
        [r["answer_response_mask"] for r in records], batch_first=True, padding_value=0.0
    )

    # batch["old_log_probs"] = old_log_probs
    batch["advantages"] = advantages
    batch["response_mask"] = response_mask
    batch["think_response_mask"] = think_response_mask
    batch["answer_response_mask"] = answer_response_mask

    if "ref_log_probs" in records[0]:
        batch["ref_log_probs"] = pad_sequence(
            [r["ref_log_probs"] for r in records], batch_first=True, padding_value=0.0
        )
    return batch


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def _release_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_completion_masks_from_batch(
    batch: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["completion_mask"][:, 1:].float(),
        batch["think_completion_mask"][:, 1:].float(),
        batch["answer_completion_mask"][:, 1:].float(),
    )


def _compute_logprobs_microbatched(
    model: torch.nn.Module,
    encoded_samples: List[Dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
    temperature: float,
    micro_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if micro_batch_size <= 0:
        raise ValueError(f"logprob micro-batch size must be > 0, got {micro_batch_size}.")

    ref_logprob_rows: List[torch.Tensor] = []
    response_mask_rows: List[torch.Tensor] = []
    think_response_mask_rows: List[torch.Tensor] = []
    answer_response_mask_rows: List[torch.Tensor] = []

    for start in range(0, len(encoded_samples), micro_batch_size):
        chunk_samples = encoded_samples[start : start + micro_batch_size]
        chunk_batch = _collate_mm_samples(chunk_samples, pad_token_id=pad_token_id)
        chunk_batch = _to_device(chunk_batch, device)
        (
            chunk_ref_log_probs,
            chunk_response_mask,
            chunk_think_response_mask,
            chunk_answer_response_mask,
        ) = _compute_completion_logprobs(model, chunk_batch, temperature=temperature)

        for row_idx, sample in enumerate(chunk_samples):
            log_len = sample["input_ids"].numel() - 1
            ref_logprob_rows.append(chunk_ref_log_probs[row_idx, :log_len].detach().cpu())
            response_mask_rows.append(chunk_response_mask[row_idx, :log_len].detach().cpu())
            think_response_mask_rows.append(chunk_think_response_mask[row_idx, :log_len].detach().cpu())
            answer_response_mask_rows.append(chunk_answer_response_mask[row_idx, :log_len].detach().cpu())

        del (
            chunk_batch,
            chunk_ref_log_probs,
            chunk_response_mask,
            chunk_think_response_mask,
            chunk_answer_response_mask,
        )

    return (
        pad_sequence(ref_logprob_rows, batch_first=True, padding_value=0.0),
        pad_sequence(response_mask_rows, batch_first=True, padding_value=0.0),
        pad_sequence(think_response_mask_rows, batch_first=True, padding_value=0.0),
        pad_sequence(answer_response_mask_rows, batch_first=True, padding_value=0.0),
    )

# Compute log probabilities in chunks instead of materializing the full
# log-softmax tensor at once.
def _compute_completion_logprobs(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs = {}
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "image_branch_flags"):
        if key in batch:
            kwargs[key] = batch[key]

    labels = batch["input_ids"][:, 1:]
    response_mask = batch["completion_mask"][:, 1:].float()
    think_response_mask = batch["think_completion_mask"][:, 1:].float()
    answer_response_mask = batch["answer_completion_mask"][:, 1:].float()
    token_log_probs = torch.zeros_like(response_mask, dtype=torch.float32)

    active_positions = response_mask.nonzero(as_tuple=False)
    logits_offset = 0
    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "use_cache": False,
        **kwargs,
    }
    if active_positions.numel() > 0 and _supports_model_forward_kwarg(model, "logits_to_keep"):
        first_response_pos = int(active_positions[:, 1].min().item())
        logits_to_keep = max(1, labels.size(1) - first_response_pos)
        # Keep one extra position so the post-forward shift still lines up with labels.
        model_inputs["logits_to_keep"] = logits_to_keep + 1
        logits_offset = labels.size(1) - logits_to_keep

    outputs = model(**model_inputs)
    logits = outputs.logits[:, :-1, :]
    if logits_offset > 0:
        logits = logits[:, - (labels.size(1) - logits_offset) :, :]
    if temperature and temperature > 0 and temperature != 1.0:
        logits.div_(temperature)

    # TRL and similar RLHF trainers avoid materializing a full float32
    # [batch, seq, vocab] logprob tensor. We only need completion-token
    # logprobs, so compute them in small row chunks.
    chunk_size = 128
    for batch_idx in range(logits.size(0)):
        sample_active_positions = response_mask[batch_idx].nonzero(as_tuple=False).flatten()
        if sample_active_positions.numel() == 0:
            continue
        sample_labels = labels[batch_idx, sample_active_positions]
        for pos_chunk, label_chunk in zip(
            sample_active_positions.split(chunk_size),
            sample_labels.split(chunk_size),
        ):
            shifted_pos_chunk = pos_chunk - logits_offset
            logits_chunk = logits[batch_idx, shifted_pos_chunk, :]
            log_probs_chunk = torch.nn.functional.log_softmax(
                logits_chunk,
                dim=-1,
                dtype=torch.float32,
            )
            token_log_probs[batch_idx, pos_chunk] = log_probs_chunk.gather(
                dim=-1,
                index=label_chunk.unsqueeze(-1),
            ).squeeze(-1)

    del outputs, logits
    token_log_probs = token_log_probs * response_mask
    return token_log_probs, response_mask, think_response_mask, answer_response_mask


def _build_token_level_rewards(reward_scores: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    #rewards = reward_scores.unsqueeze(-1).to(dtype=torch.float32) * response_mask.to(dtype=torch.float32)

    mask = response_mask.to(dtype=torch.float32)
    token_counts = mask.sum(dim=-1, keepdim=True)  # shape: [B, 1]
    token_counts = token_counts.clamp(min=1.0)
    rewards = reward_scores.unsqueeze(-1).to(torch.float32) / token_counts
    rewards = rewards * mask
    return rewards


def _trainer_state_path(checkpoint_dir: str, rank: Optional[int] = None) -> str:
    rank = _get_rank() if rank is None else int(rank)
    return os.path.join(checkpoint_dir, f"trainer_state_rank{rank}.pt")


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_device"] = torch.cuda.current_device()
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def _restore_rng_state(state: Dict[str, Any]):
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    cuda_rng_state = state.get("cuda_rng_state")
    if torch.cuda.is_available() and cuda_rng_state is not None:
        cuda_device = int(state.get("cuda_device", torch.cuda.current_device()))
        torch.cuda.set_device(cuda_device)
        torch.cuda.set_rng_state(cuda_rng_state, device=cuda_device)


def _torch_load_full_state(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_trainer_state(
    checkpoint_dir: str,
    global_step: int,
    epoch: int,
    batch_idx: int,
    dataset_size: int,
    batch_size: int,
    seed: int,
    dataset_use: str,
    max_prompt_samples: int,
    rollout_num_workers: int,
    enable_local_image: bool,
    global_sample_cursor: Optional[int] = None,
    data_reference_world_size: Optional[int] = None,
    data_reference_batch_size: Optional[int] = None,
    warm_start_source_checkpoint: Optional[str] = None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if global_sample_cursor is None:
        global_sample_cursor = (
            int(batch_idx) * int(batch_size) * max(1, int(_get_world_size()))
        )
    state = {
        "global_step": int(global_step),
        # These point to the next prompt batch to process.
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "global_sample_cursor": int(global_sample_cursor),
        "rank": int(_get_rank()),
        "world_size": int(_get_world_size()),
        "dataset_size": int(dataset_size),
        "batch_size": int(batch_size),
        "data_reference_world_size": int(
            data_reference_world_size
            if data_reference_world_size is not None
            else _get_world_size()
        ),
        "data_reference_batch_size": int(
            data_reference_batch_size
            if data_reference_batch_size is not None
            else batch_size
        ),
        "seed": int(seed),
        "dataset_use": str(dataset_use),
        "max_prompt_samples": int(max_prompt_samples),
        "rollout_num_workers": int(rollout_num_workers),
        "enable_local_image": bool(enable_local_image),
        "warm_start_source_checkpoint": (
            str(warm_start_source_checkpoint) if warm_start_source_checkpoint else None
        ),
        "rng_state": _capture_rng_state(),
    }
    torch.save(state, _trainer_state_path(checkpoint_dir))


def _load_trainer_state(
    checkpoint_dir: str,
    dataset_size: int,
    batch_size: int,
    seed: int,
    dataset_use: str,
    max_prompt_samples: int,
    rollout_num_workers: int,
    enable_local_image: bool,
) -> Dict[str, Any]:
    state_path = _trainer_state_path(checkpoint_dir)
    if not os.path.isfile(state_path):
        raise RuntimeError(
            f"Missing rank-local trainer state: {state_path}. "
            "This looks like a weight-only or legacy checkpoint, not a full exact-resume checkpoint."
        )

    state = _torch_load_full_state(state_path)
    saved_world_size = int(state.get("world_size", -1))
    if saved_world_size != _get_world_size():
        raise RuntimeError(
            "Cannot resume exactly with a different world size: "
            f"checkpoint world_size={saved_world_size}, current world_size={_get_world_size()}."
        )
    saved_rank = int(state.get("rank", -1))
    if saved_rank != _get_rank():
        raise RuntimeError(
            f"Loaded wrong trainer state for rank {_get_rank()}: file contains rank={saved_rank}."
        )
    saved_dataset_size = int(state.get("dataset_size", -1))
    if saved_dataset_size != int(dataset_size):
        raise RuntimeError(
            "Cannot resume exactly because prompt dataset size changed: "
            f"checkpoint dataset_size={saved_dataset_size}, current dataset_size={dataset_size}."
        )
    saved_batch_size = int(state.get("batch_size", -1))
    if saved_batch_size != int(batch_size):
        raise RuntimeError(
            "Cannot resume exactly with a different per-device train batch size: "
            f"checkpoint batch_size={saved_batch_size}, current batch_size={batch_size}."
        )
    saved_seed = int(state.get("seed", -1))
    if saved_seed != int(seed):
        raise RuntimeError(
            f"Cannot resume exactly with a different seed: checkpoint seed={saved_seed}, current seed={seed}."
        )
    saved_dataset_use = str(state.get("dataset_use", ""))
    if saved_dataset_use != str(dataset_use):
        raise RuntimeError(
            "Cannot resume exactly with a different dataset_use: "
            f"checkpoint dataset_use={saved_dataset_use!r}, current dataset_use={dataset_use!r}."
        )
    saved_max_prompt_samples = int(state.get("max_prompt_samples", -1))
    if saved_max_prompt_samples != int(max_prompt_samples):
        raise RuntimeError(
            "Cannot resume exactly with a different max_prompt_samples: "
            f"checkpoint max_prompt_samples={saved_max_prompt_samples}, current max_prompt_samples={max_prompt_samples}."
        )
    saved_rollout_num_workers = int(state.get("rollout_num_workers", -1))
    if saved_rollout_num_workers != int(rollout_num_workers):
        raise RuntimeError(
            "Cannot resume exactly with a different rollout_num_workers: "
            f"checkpoint rollout_num_workers={saved_rollout_num_workers}, "
            f"current rollout_num_workers={rollout_num_workers}."
        )
    saved_enable_local_image = bool(state.get("enable_local_image", False))
    if saved_enable_local_image != bool(enable_local_image):
        raise RuntimeError(
            "Cannot resume exactly with a different enable_local_image setting: "
            f"checkpoint enable_local_image={saved_enable_local_image}, "
            f"current enable_local_image={bool(enable_local_image)}."
        )
    return state


def _validate_data_progress_state(
    state: Dict[str, Any],
    *,
    dataset_size: int,
    seed: int,
    dataset_use: str,
    max_prompt_samples: int,
    rollout_num_workers: int,
    enable_local_image: bool,
    context: str,
) -> None:
    expected_values = {
        "dataset_size": int(dataset_size),
        "seed": int(seed),
        "dataset_use": str(dataset_use),
        "max_prompt_samples": int(max_prompt_samples),
        "rollout_num_workers": int(rollout_num_workers),
        "enable_local_image": bool(enable_local_image),
    }
    for key, expected in expected_values.items():
        if key not in state:
            raise RuntimeError(f"{context} is missing required data metadata: {key}.")
        saved = state[key]
        if isinstance(expected, bool):
            matches = bool(saved) == expected
        elif isinstance(expected, int):
            matches = int(saved) == expected
        else:
            matches = str(saved) == expected
        if not matches:
            raise RuntimeError(
                f"{context} data metadata mismatch for {key}: saved={saved!r}, current={expected!r}."
            )


def _resolve_resume_checkpoint(training_args: TrainingArguments) -> Optional[str]:
    raw_value = getattr(training_args, "resume_from_checkpoint", None)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        if not raw_value:
            return None
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None:
            raise RuntimeError(
                f"--resume_from_checkpoint was set, but no checkpoint-* directory was found in {training_args.output_dir}."
            )
        return last_checkpoint

    resume_checkpoint = str(raw_value).strip()
    if not resume_checkpoint:
        return None
    if resume_checkpoint.lower() in {"false", "0", "no"}:
        return None
    if resume_checkpoint.lower() in {"true", "1", "yes"}:
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None:
            raise RuntimeError(
                f"--resume_from_checkpoint was set, but no checkpoint-* directory was found in {training_args.output_dir}."
            )
        return last_checkpoint
    return resume_checkpoint


def _build_epoch_batches(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    epoch: int,
    seed: int,
    world_size: int,
    rank: int,
    drop_last: bool = False,
    reference_world_size: Optional[int] = None,
    reference_batch_size: Optional[int] = None,
) -> List[List[int]]:
    if reference_world_size is not None or reference_batch_size is not None:
        if not drop_last:
            raise ValueError(
                "World-size-independent data continuation currently requires dataloader_drop_last=True."
            )
        reference_world_size = int(reference_world_size or world_size)
        reference_batch_size = int(reference_batch_size or batch_size)
        if reference_world_size <= 0 or reference_batch_size <= 0:
            raise ValueError("Reference world size and batch size must be positive.")

        # Reconstruct the exact global permutation that DistributedSampler used
        # under the source topology, including its padding behavior.
        generator = torch.Generator()
        generator.manual_seed(int(seed) + int(epoch))
        dataset_size = len(dataset)
        global_indices = torch.randperm(dataset_size, generator=generator).tolist()
        reference_num_samples = int(math.ceil(dataset_size / reference_world_size))
        reference_total_size = reference_num_samples * reference_world_size
        padding_size = reference_total_size - len(global_indices)
        if padding_size > 0:
            if padding_size <= len(global_indices):
                global_indices += global_indices[:padding_size]
            else:
                repeats = int(math.ceil(padding_size / max(1, len(global_indices))))
                global_indices += (global_indices * repeats)[:padding_size]

        # Match the source run's rank-local drop_last boundary. This matters
        # because changing 8 GPUs to 4 GPUs otherwise admits 32 extra prompts
        # at the end of this particular MIMIC_COT epoch.
        reference_full_batches = reference_num_samples // reference_batch_size
        reference_usable_size = (
            reference_full_batches * reference_batch_size * reference_world_size
        )
        global_indices = global_indices[:reference_usable_size]

        current_global_batch_size = int(batch_size) * int(world_size)
        if reference_usable_size % current_global_batch_size != 0:
            raise ValueError(
                "The source epoch boundary cannot be represented by full batches under the current "
                f"topology: source_usable={reference_usable_size}, "
                f"current_global_batch={current_global_batch_size}."
            )

        batches: List[List[int]] = []
        for start in range(0, reference_usable_size, current_global_batch_size):
            global_batch = global_indices[start : start + current_global_batch_size]
            local_batch = global_batch[int(rank) :: int(world_size)]
            if len(local_batch) != int(batch_size):
                raise RuntimeError(
                    f"Internal global sampler error: rank={rank} received {len(local_batch)} "
                    f"indices, expected {batch_size}."
                )
            batches.append(local_batch)
        return batches

    sampler = DistributedSampler(
        dataset,
        num_replicas=max(1, int(world_size)),
        rank=int(rank),
        shuffle=True,
        seed=int(seed),
        drop_last=False,
    )
    sampler.set_epoch(int(epoch))
    indices = list(iter(sampler))
    batches = [
        indices[start : start + int(batch_size)]
        for start in range(0, len(indices), int(batch_size))
    ]
    if drop_last:
        batches = [batch for batch in batches if len(batch) == int(batch_size)]
    return batches


def _save_hf_artifacts(model: torch.nn.Module, processor: Any, output_dir: str):
    model.config.save_pretrained(output_dir)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def _save_checkpoint(
    model: torch.nn.Module,
    processor: Any,
    output_dir: str,
    step: int,
    epoch: int,
    batch_idx: int,
    dataset_size: int,
    batch_size: int,
    seed: int,
    dataset_use: str,
    max_prompt_samples: int,
    rollout_num_workers: int,
    enable_local_image: bool,
    global_sample_cursor: Optional[int] = None,
    data_reference_world_size: Optional[int] = None,
    data_reference_batch_size: Optional[int] = None,
    warm_start_source_checkpoint: Optional[str] = None,
):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    if hasattr(model, "save_checkpoint") and hasattr(model, "module") and not _is_peft_model(model_to_save):
        client_state = {
            "global_step": int(step),
            "epoch": int(epoch),
            "batch_idx": int(batch_idx),
        }
        _barrier()
        if _is_main_process():
            _save_hf_artifacts(model.module, processor, ckpt_dir)
        if hasattr(model, "save_16bit_model"):
            model.save_16bit_model(ckpt_dir, save_filename="pytorch_model.bin")
        model.save_checkpoint(
            ckpt_dir,
            tag=DEEPSPEED_CHECKPOINT_TAG,
            client_state=client_state,
        )
        _save_trainer_state(
            ckpt_dir,
            global_step=step,
            epoch=epoch,
            batch_idx=batch_idx,
            dataset_size=dataset_size,
            batch_size=batch_size,
            seed=seed,
            dataset_use=dataset_use,
            max_prompt_samples=max_prompt_samples,
            rollout_num_workers=rollout_num_workers,
            enable_local_image=enable_local_image,
            global_sample_cursor=global_sample_cursor,
            data_reference_world_size=data_reference_world_size,
            data_reference_batch_size=data_reference_batch_size,
            warm_start_source_checkpoint=warm_start_source_checkpoint,
        )
        _barrier()
        return

    _barrier()
    if _is_main_process():
        _save_hf_artifacts(model_to_save, processor, ckpt_dir)
        model_to_save.save_pretrained(ckpt_dir)
    _save_trainer_state(
        ckpt_dir,
        global_step=step,
        epoch=epoch,
        batch_idx=batch_idx,
        dataset_size=dataset_size,
        batch_size=batch_size,
        seed=seed,
        dataset_use=dataset_use,
        max_prompt_samples=max_prompt_samples,
        rollout_num_workers=rollout_num_workers,
        enable_local_image=enable_local_image,
        global_sample_cursor=global_sample_cursor,
        data_reference_world_size=data_reference_world_size,
        data_reference_batch_size=data_reference_batch_size,
        warm_start_source_checkpoint=warm_start_source_checkpoint,
    )
    _barrier()


def _rotate_checkpoints(output_dir: str, save_total_limit: Optional[int]) -> None:
    if save_total_limit is None or int(save_total_limit) <= 0:
        return
    _barrier()
    if _is_main_process():
        checkpoints: List[tuple[int, Path]] = []
        root = Path(output_dir)
        if root.is_dir():
            for candidate in root.iterdir():
                match = re.fullmatch(r"checkpoint-(\d+)", candidate.name)
                if match and candidate.is_dir():
                    checkpoints.append((int(match.group(1)), candidate))
        checkpoints.sort(key=lambda item: item[0])
        excess = len(checkpoints) - int(save_total_limit)
        for _, checkpoint_path in checkpoints[: max(0, excess)]:
            _print_main(f"[checkpoint] removing old checkpoint: {checkpoint_path}")
            shutil.rmtree(checkpoint_path)
    _barrier()


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GRPOArguments, VerlBridgeArguments)
    )
    model_args, data_args, training_args, grpo_args, bridge_args = parser.parse_args_into_dataclasses()

    if int(training_args.per_device_train_batch_size) <= 0:
        raise ValueError("per_device_train_batch_size must be > 0.")
    if int(training_args.gradient_accumulation_steps) <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0.")
    if int(grpo_args.num_generations) <= 0:
        raise ValueError("num_generations must be > 0.")
    if int(grpo_args.max_completion_length) <= 0:
        raise ValueError("max_completion_length must be > 0.")
    if int(grpo_args.minibatch_size) <= 0:
        raise ValueError("minibatch_size must be > 0.")
    if int(grpo_args.inner_epochs) <= 0:
        raise ValueError("inner_epochs must be > 0.")
    if float(grpo_args.temperature) <= 0:
        raise ValueError("temperature must be > 0 for GRPO sampling/log-probability alignment.")
    if not (0.0 < float(grpo_args.top_p) <= 1.0):
        raise ValueError("top_p must be in (0, 1].")
    if float(grpo_args.reward_eps) <= 0:
        raise ValueError("reward_eps must be > 0.")

    _seed_everything(training_args.seed)
    os.makedirs(training_args.output_dir, exist_ok=True)
    device, rank, world_size = _setup_runtime(training_args)
    use_deepspeed = training_args.deepspeed is not None
    if not use_deepspeed and int(training_args.gradient_accumulation_steps) != 1:
        raise ValueError(
            "This custom non-DeepSpeed loop does not implement gradient accumulation; "
            "set gradient_accumulation_steps=1 or enable DeepSpeed."
        )
    resume_checkpoint = _resolve_resume_checkpoint(training_args)
    if resume_checkpoint is not None and not use_deepspeed:
        raise NotImplementedError("Exact resume_from_checkpoint is currently implemented for DeepSpeed training only.")
    raw_warm_start = str(bridge_args.warm_start_from_checkpoint or "").strip()
    warm_start_checkpoint = (
        None
        if raw_warm_start.lower() in {"", "false", "0", "no", "none"}
        else os.path.abspath(raw_warm_start)
    )
    if resume_checkpoint is not None and warm_start_checkpoint is not None:
        raise ValueError(
            "--resume_from_checkpoint and --warm_start_from_checkpoint are mutually exclusive. "
            "Use exact resume only for an unchanged ZeRO topology."
        )

    data_progress_preview: Optional[Dict[str, Any]] = None
    if warm_start_checkpoint is not None:
        if not os.path.isdir(warm_start_checkpoint):
            raise RuntimeError(
                f"Warm-start checkpoint does not exist: {warm_start_checkpoint}"
            )
        warm_model_path = os.path.join(warm_start_checkpoint, "pytorch_model.bin")
        if not os.path.isfile(warm_model_path):
            raise RuntimeError(
                f"Warm-start checkpoint is missing consolidated policy weights: {warm_model_path}"
            )
        source_state_path = _trainer_state_path(warm_start_checkpoint, rank=0)
        if not os.path.isfile(source_state_path):
            raise RuntimeError(
                f"Warm-start checkpoint is missing source data progress: {source_state_path}"
            )
        data_progress_preview = _torch_load_full_state(source_state_path)
        model_args.model_name_or_path = warm_start_checkpoint
    elif resume_checkpoint is not None:
        preview_path = _trainer_state_path(resume_checkpoint)
        if not os.path.isfile(preview_path):
            raise RuntimeError(f"Missing rank-local trainer state: {preview_path}")
        data_progress_preview = _torch_load_full_state(preview_path)

    _print_main(
        f"[init] rank={rank}/{world_size}, deepspeed={use_deepspeed}, "
        f"output_dir={training_args.output_dir}, dataset={data_args.dataset_use}"
    )
    if resume_checkpoint is not None:
        _print_main(f"[init] exact resume requested from {resume_checkpoint}")
    if warm_start_checkpoint is not None:
        _print_main(
            f"[init] four-GPU warm start requested from {warm_start_checkpoint}; "
            "policy weights/data cursor will continue, optimizer and ZeRO state will be rebuilt"
        )
    tb_writer = _create_tensorboard_writer(training_args.output_dir)
    _print_main(f"[init] loading policy model from {model_args.model_name_or_path}")
    # The full-overlay loader restores decoder, global vision, and local vision
    # tensors while streaming checkpoint shards. During warm start it must read
    # from the consolidated GRPO checkpoint itself; otherwise the original SFT
    # overlay would overwrite the warm checkpoint's language weights after
    # from_pretrained().
    policy_overlay_path = (
        warm_start_checkpoint
        if warm_start_checkpoint is not None
        else model_args.local_branch_model_path
    )
    model, processor = _load_policy_model(
        model_args,
        data_args,
        training_args,
        local_branch_model_path=policy_overlay_path,
        local_branch_strict_load=model_args.local_branch_strict_load,
    )
    if training_args.lora_enable:
        model = _enable_lora(model, model_args, training_args)
    _enable_gradient_checkpointing(model, training_args)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is required for variable-length GRPO collation.")
    if not use_deepspeed:
        model.to(device)
    model.train()

    ref_model = None
    if grpo_args.ref_model_name_or_path:
        _print_main(f"[init] loading reference model from {grpo_args.ref_model_name_or_path}")
        ref_model, _ = _load_policy_model(
            ModelArguments(
                model_name_or_path=grpo_args.ref_model_name_or_path,
                use_local_vision_branch=model_args.use_local_vision_branch,
                tune_mm_local_vision=model_args.tune_mm_local_vision,
                tune_mm_local_mlp=model_args.tune_mm_local_mlp,
                local_branch_model_path=model_args.local_branch_model_path,
                local_branch_strict_load=model_args.local_branch_strict_load,
            ),
            data_args,
            training_args,
            local_branch_model_path=model_args.local_branch_model_path,
            local_branch_strict_load=model_args.local_branch_strict_load,
        )
        if not use_deepspeed:
            ref_model.to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    _print_main("[init] building prompt dataset")
    prompt_dataset = GRPOPromptDataset(
        PromptDatasetConfig(
            dataset_use=data_args.dataset_use,
            enable_local_image=getattr(data_args, "enable_local_image", False),
            shuffle=True,
            max_prompt_samples=bridge_args.max_prompt_samples,
            seed=training_args.seed,
        )
    )
    _print_main(f"[init] prompt dataset size={len(prompt_dataset)}")
    if len(prompt_dataset) == 0:
        raise ValueError("Prompt dataset is empty after conversion.")

    per_device_batch_size = int(training_args.per_device_train_batch_size)
    data_reference_world_size = int(world_size)
    data_reference_batch_size = int(per_device_batch_size)
    data_progress_global_cursor = 0
    data_progress_start_batch_idx = 0
    use_reference_data_sampler = False
    if data_progress_preview is not None:
        progress_context = (
            f"warm-start checkpoint {warm_start_checkpoint}"
            if warm_start_checkpoint is not None
            else f"resume checkpoint {resume_checkpoint}"
        )
        _validate_data_progress_state(
            data_progress_preview,
            dataset_size=len(prompt_dataset),
            seed=training_args.seed,
            dataset_use=data_args.dataset_use,
            max_prompt_samples=bridge_args.max_prompt_samples,
            rollout_num_workers=bridge_args.rollout_num_workers,
            enable_local_image=getattr(data_args, "enable_local_image", False),
            context=progress_context,
        )
        source_world_size = int(data_progress_preview.get("world_size", -1))
        source_batch_size = int(data_progress_preview.get("batch_size", -1))
        source_batch_idx = int(data_progress_preview.get("batch_idx", -1))
        if source_world_size <= 0 or source_batch_size <= 0 or source_batch_idx < 0:
            raise RuntimeError(
                f"{progress_context} contains invalid source topology/data cursor: "
                f"world_size={source_world_size}, batch_size={source_batch_size}, "
                f"batch_idx={source_batch_idx}."
            )

        data_reference_world_size = int(
            data_progress_preview.get("data_reference_world_size", source_world_size)
        )
        data_reference_batch_size = int(
            data_progress_preview.get("data_reference_batch_size", source_batch_size)
        )
        inferred_global_cursor = (
            source_batch_idx * source_batch_size * source_world_size
        )
        data_progress_global_cursor = int(
            data_progress_preview.get("global_sample_cursor", inferred_global_cursor)
        )
        current_global_batch_size = int(world_size) * per_device_batch_size
        if data_progress_global_cursor % current_global_batch_size != 0:
            raise RuntimeError(
                f"{progress_context} global sample cursor={data_progress_global_cursor} "
                f"is not aligned to current global batch size={current_global_batch_size}."
            )
        data_progress_start_batch_idx = (
            data_progress_global_cursor // current_global_batch_size
        )
        use_reference_data_sampler = bool(
            warm_start_checkpoint is not None
            or "data_reference_world_size" in data_progress_preview
            or "data_reference_batch_size" in data_progress_preview
        )

    epoch0_batches = _build_epoch_batches(
        prompt_dataset,
        batch_size=per_device_batch_size,
        epoch=0,
        seed=training_args.seed,
        world_size=world_size,
        rank=rank,
        drop_last=bool(getattr(training_args, "dataloader_drop_last", False)),
        reference_world_size=(
            data_reference_world_size if use_reference_data_sampler else None
        ),
        reference_batch_size=(
            data_reference_batch_size if use_reference_data_sampler else None
        ),
    )
    batches_per_epoch = len(epoch0_batches)
    if batches_per_epoch <= 0:
        raise ValueError(
            "No prompt batches remain after distributed sharding/drop_last. "
            "Reduce per_device_train_batch_size or disable dataloader_drop_last."
        )
    if data_progress_start_batch_idx > batches_per_epoch:
        raise RuntimeError(
            f"Data cursor maps to batch_idx={data_progress_start_batch_idx}, "
            f"beyond the strict epoch batch count={batches_per_epoch}."
        )
    if data_progress_preview is not None:
        _print_main(
            "[data-resume] "
            f"source_world_size={int(data_progress_preview['world_size'])}, "
            f"source_batch_size={int(data_progress_preview['batch_size'])}, "
            f"global_sample_cursor={data_progress_global_cursor}, "
            f"current_batch_idx={data_progress_start_batch_idx}, "
            f"reference_epoch_batches={batches_per_epoch}"
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check tune_mm_* flags.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )
    if use_deepspeed:
        _print_main("[init] initializing DeepSpeed for policy model")
        model, optimizer, _ = _init_deepspeed_engine(
            model=model,
            optimizer=optimizer,
            training_args=training_args,
            train_micro_batch_size=grpo_args.minibatch_size,
        )
        if ref_model is not None:
            _print_main("[init] initializing DeepSpeed for reference model")
            ref_model, _, _ = _init_deepspeed_engine(
                model=ref_model,
                optimizer=None,
                training_args=training_args,
                train_micro_batch_size=training_args.per_device_train_batch_size,
            )
            ref_model.eval()

    start_global_step = 0
    start_epoch = 0
    start_batch_idx = 0
    resume_trainer_state: Optional[Dict[str, Any]] = None
    if resume_checkpoint is not None:
        if not os.path.isdir(resume_checkpoint):
            raise RuntimeError(f"Resume checkpoint does not exist or is not a directory: {resume_checkpoint}")
        ds_state_dir = os.path.join(resume_checkpoint, DEEPSPEED_CHECKPOINT_TAG)
        if not os.path.isdir(ds_state_dir):
            raise RuntimeError(
                f"Missing DeepSpeed checkpoint tag directory: {ds_state_dir}. "
                "This checkpoint cannot be used for exact training resume."
            )
        load_path, client_state = model.load_checkpoint(
            resume_checkpoint,
            tag=DEEPSPEED_CHECKPOINT_TAG,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if load_path is None:
            raise RuntimeError(f"DeepSpeed failed to load checkpoint from {resume_checkpoint}.")

        resume_trainer_state = _load_trainer_state(
            resume_checkpoint,
            dataset_size=len(prompt_dataset),
            batch_size=per_device_batch_size,
            seed=training_args.seed,
            dataset_use=data_args.dataset_use,
            max_prompt_samples=bridge_args.max_prompt_samples,
            rollout_num_workers=bridge_args.rollout_num_workers,
            enable_local_image=getattr(data_args, "enable_local_image", False),
        )
        start_global_step = int(resume_trainer_state["global_step"])
        start_epoch = int(resume_trainer_state["epoch"])
        start_batch_idx = int(resume_trainer_state["batch_idx"])
        if start_batch_idx != data_progress_start_batch_idx:
            raise RuntimeError(
                "Exact-resume batch index disagrees with the saved global sample cursor: "
                f"batch_idx={start_batch_idx}, cursor_batch_idx={data_progress_start_batch_idx}."
            )

        client_state = client_state or {}
        for key, expected in (
            ("global_step", start_global_step),
            ("epoch", start_epoch),
            ("batch_idx", start_batch_idx),
        ):
            if key in client_state and int(client_state[key]) != int(expected):
                raise RuntimeError(
                    f"Resume metadata mismatch for {key}: DeepSpeed client_state={client_state[key]}, "
                    f"rank-local trainer_state={expected}."
                )
        _print_main(
            f"[resume] loaded checkpoint path={load_path}, "
            f"global_step={start_global_step}, epoch={start_epoch}, batch_idx={start_batch_idx}"
        )
    elif warm_start_checkpoint is not None:
        if data_progress_preview is None:
            raise RuntimeError("Internal error: warm-start data progress was not loaded.")
        start_global_step = int(data_progress_preview["global_step"])
        start_epoch = int(data_progress_preview["epoch"])
        start_batch_idx = int(data_progress_start_batch_idx)
        _print_main(
            "[warm-start] loaded consolidated policy weights with a fresh optimizer/ZeRO state; "
            f"source_global_step={start_global_step}, epoch={start_epoch}, "
            f"global_sample_cursor={data_progress_global_cursor}, "
            f"four_gpu_batch_idx={start_batch_idx}"
        )

    _print_main(
        f"[init] creating rollout backend "
        f"{bridge_args.rollout_backend_module}.{bridge_args.rollout_backend_class}"
    )
    rollout_cls = load_rollout_backend(bridge_args.rollout_backend_module, bridge_args.rollout_backend_class)
    rollout_backend = rollout_cls(
        model_name_or_path=grpo_args.vllm_model_name_or_path,
        policy_model=model,
        processor=processor,
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        grpo_args=grpo_args,
    )
    _print_main("[init] loading reward function")
    reward_fn = load_reward_function(bridge_args.reward_module, bridge_args.reward_function_name)
    reward_batch_fn = load_reward_batch_function(bridge_args.reward_module)

    # Bootstrap vLLM from the model actually loaded by the trainer before the
    # first sample.  Relying on a manually exported directory can make the
    # first rollout off-policy when it does not exactly match the SFT overlay.
    if hasattr(rollout_backend, "maybe_sync_from_policy_model"):
        ensure_runtime = getattr(rollout_backend, "_ensure_runtime", None)
        if callable(ensure_runtime):
            runtime_error: Optional[str] = None
            try:
                ensure_runtime()
            except Exception as exc:
                runtime_error = (
                    f"rank={rank} {type(exc).__name__}: {exc}\n"
                    + traceback.format_exc(limit=20)
                )

            runtime_errors: List[Optional[str]] = [runtime_error]
            if _is_dist_initialized():
                runtime_errors = [None for _ in range(world_size)]
                # Shared vLLM owns collocated CUDA/NCCL resources after the
                # owner creates it.  Use its CPU/Gloo control group for status
                # exchange instead of starting a needless NCCL object collective.
                runtime_group = getattr(rollout_backend, "rollout_control_group", None)
                dist.all_gather_object(runtime_errors, runtime_error, group=runtime_group)
            runtime_failures = [error for error in runtime_errors if error]
            if runtime_failures:
                raise RuntimeError(
                    "Rollout runtime initialization failed collectively:\n"
                    + "\n".join(runtime_failures)
                )
        rollout_backend.maybe_sync_from_policy_model(model, start_global_step)

    if resume_trainer_state is not None:
        _restore_rng_state(resume_trainer_state["rng_state"])
        _print_main("[resume] restored rank-local RNG state")
    elif warm_start_checkpoint is not None:
        _print_main(
            "[warm-start] rank-local RNG is intentionally reinitialized from the current seed; "
            "the prompt cursor is preserved, but stochastic rollout trajectories are not exact-resume equivalent"
        )

    num_train_epochs = max(1, int(math.ceil(training_args.num_train_epochs)))
    max_steps_enabled = bool(training_args.max_steps and training_args.max_steps > 0)
    if max_steps_enabled:
        total_steps = int(training_args.max_steps)
    else:
        remaining_current_epoch = (
            max(0, batches_per_epoch - start_batch_idx)
            if start_epoch < num_train_epochs
            else 0
        )
        remaining_future_epochs = max(0, num_train_epochs - start_epoch - 1)
        remaining_updates = (
            remaining_current_epoch + remaining_future_epochs * batches_per_epoch
        )
        total_steps = max(start_global_step, start_global_step + remaining_updates)
    _print_main(
        f"[init] total_steps={total_steps}, logging_steps={training_args.logging_steps}, "
        f"max_steps_overrides_epochs={int(max_steps_enabled)}"
    )

    global_step = start_global_step
    next_resume_epoch = start_epoch
    next_resume_batch_idx = start_batch_idx
    next_resume_global_sample_cursor = int(data_progress_global_cursor)
    max_consecutive_dropped_batches = int(
        os.environ.get("QWENVL_GRPO_MAX_CONSECUTIVE_DROPPED_BATCHES", "20")
    )
    if max_consecutive_dropped_batches <= 0:
        raise ValueError("QWENVL_GRPO_MAX_CONSECUTIVE_DROPPED_BATCHES must be > 0.")
    consecutive_dropped_batches = 0
    epoch = start_epoch
    while global_step < total_steps and (max_steps_enabled or epoch < num_train_epochs):
        epoch_batches = _build_epoch_batches(
            prompt_dataset,
            batch_size=per_device_batch_size,
            epoch=epoch,
            seed=training_args.seed,
            world_size=world_size,
            rank=rank,
            drop_last=bool(getattr(training_args, "dataloader_drop_last", False)),
            reference_world_size=(
                data_reference_world_size if use_reference_data_sampler else None
            ),
            reference_batch_size=(
                data_reference_batch_size if use_reference_data_sampler else None
            ),
        )
        num_epoch_batches = len(epoch_batches)
        skip_batches = start_batch_idx if epoch == start_epoch else 0
        if skip_batches > num_epoch_batches:
            raise RuntimeError(
                f"Resume batch_idx={skip_batches} is beyond epoch {epoch} batch count={num_epoch_batches}."
            )
        if skip_batches == num_epoch_batches:
            next_resume_epoch = epoch + 1
            next_resume_batch_idx = 0
            next_resume_global_sample_cursor = 0
            epoch += 1
            continue
        if skip_batches > 0:
            epoch_batches = epoch_batches[skip_batches:]

        dataloader = DataLoader(
            prompt_dataset,
            batch_sampler=epoch_batches,
            collate_fn=lambda x: x,
            num_workers=bridge_args.rollout_num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for batch_offset, prompt_batch in enumerate(dataloader):
            batch_idx = skip_batches + batch_offset
            if global_step >= total_steps:
                break

            requests = [
                RolloutRequest(
                    uid=sample.uid,
                    messages=sample.messages,
                    num_generations=grpo_args.num_generations,
                    max_new_tokens=grpo_args.max_completion_length,
                    temperature=grpo_args.temperature,
                    top_p=grpo_args.top_p,
                    meta=sample.meta,
                    image_branch_flags=sample.image_branch_flags,
                    anchor_image=sample.anchor_image,
                )
                for sample in prompt_batch
            ]

            if global_step == 0:
                _print_main(
                    f"[init] starting first rollout generate: batch_prompts={len(prompt_batch)}, "
                    f"num_generations={grpo_args.num_generations}"
                )
            responses = rollout_backend.generate(requests)
            if global_step == 0:
                _print_main("[init] first rollout generate finished")
            response_map = {r.uid: r for r in responses}

            encoded_samples: List[Dict[str, Any]] = []
            reward_scores: List[float] = []
            local_reward_inputs: List[Dict[str, Any]] = []
            group_index: List[str] = []
            all_completion_texts: List[str] = []

            local_unrecoverable_group = False
            for sample in prompt_batch:
                if sample.uid not in response_map:
                    print(
                        f"[warn][rank={rank}] skipping rollout batch because response is missing "
                        f"for group uid={sample.uid}.",
                        flush=True,
                    )
                    local_unrecoverable_group = True
                    break

                expected_generations = max(1, int(grpo_args.num_generations))
                completions = response_map[sample.uid].completions[:expected_generations]
                anchor_image = response_map[sample.uid].anchor_image
                group_entries: List[Optional[Dict[str, Any]]] = [None] * expected_generations
                valid_group_entries: List[Dict[str, Any]] = []
                invalid_reasons: Dict[int, str] = {}

                for generation_idx in range(expected_generations):
                    completion = completions[generation_idx] if generation_idx < len(completions) else None
                    if completion is None:
                        invalid_reasons[generation_idx] = "missing completion"
                        continue
                    text = completion.text if completion is not None else ""
                    try:
                        encoded = _encode_sample(
                            processor=processor,
                            prompt_messages=sample.messages,
                            completion_text=text,
                            anchor_image=anchor_image,
                            completion_token_ids=completion.token_ids,
                            completion_action_mask=completion.action_mask,
                            image_branch_flags=sample.image_branch_flags,
                            max_completion_length=grpo_args.max_completion_length,
                        )
                    except (InvalidRolloutTraceError, InvalidCompatibilityCompletionError) as exc:
                        invalid_reasons[generation_idx] = f"{type(exc).__name__}: {exc}"
                        continue
                    if encoded is None:
                        invalid_reasons[generation_idx] = "completion produced no trainable action tokens"
                        continue
                    effective_text = encoded.get("effective_completion_text", text)
                    entry = {
                        "encoded": encoded,
                        "reward_input": {
                            "text": effective_text,
                            "ground_truth": sample.ground_truth,
                            "meta": sample.meta,
                        },
                        "effective_text": effective_text,
                        "source_generation_idx": generation_idx,
                    }
                    group_entries[generation_idx] = entry
                    valid_group_entries.append(entry)

                if invalid_reasons and not valid_group_entries:
                    reasons = "; ".join(
                        f"generation={idx}: {reason}" for idx, reason in sorted(invalid_reasons.items())
                    )
                    print(
                        f"[warn][rank={rank}] skipping rollout batch because group uid={sample.uid} "
                        f"has no valid completion for replacement. {reasons}",
                        flush=True,
                    )
                    local_unrecoverable_group = True
                    break

                for generation_idx, reason in sorted(invalid_reasons.items()):
                    replacement = random.choice(valid_group_entries)
                    group_entries[generation_idx] = replacement
                    print(
                        f"[warn][rank={rank}] replacing invalid completion: uid={sample.uid}, "
                        f"generation={generation_idx}, "
                        f"replacement_generation={replacement['source_generation_idx']}, "
                        f"reason={reason}",
                        flush=True,
                    )

                for entry in group_entries:
                    if entry is None:
                        raise RuntimeError(
                            f"Internal completion replacement failed for group uid={sample.uid}."
                        )
                    encoded_samples.append(entry["encoded"])
                    local_reward_inputs.append(dict(entry["reward_input"]))
                    group_index.append(sample.uid)
                    all_completion_texts.append(entry["effective_text"])

            unrecoverable_group_flag = torch.tensor(
                1 if local_unrecoverable_group else 0,
                dtype=torch.int32,
                device=device,
            )
            if _is_dist_initialized():
                dist.all_reduce(unrecoverable_group_flag, op=dist.ReduceOp.MAX)
            if bool(unrecoverable_group_flag.item()):
                consecutive_dropped_batches += 1
                _print_main(
                    "[warn] discarding the current rollout batch on every rank because at least "
                    "one group had no valid completion for replacement "
                    f"(consecutive={consecutive_dropped_batches}/"
                    f"{max_consecutive_dropped_batches})."
                )
                del (
                    responses,
                    response_map,
                    encoded_samples,
                    local_reward_inputs,
                    group_index,
                    all_completion_texts,
                    unrecoverable_group_flag,
                )
                _release_cuda_cache()
                if consecutive_dropped_batches >= max_consecutive_dropped_batches:
                    raise RuntimeError(
                        "Too many consecutive rollout batches had no valid completion for "
                        "replacement. Refusing to loop forever without policy updates."
                    )
                continue
            del unrecoverable_group_flag
            consecutive_dropped_batches = 0

            if global_step == 0:
                _print_main(
                    f"[post-rollout] collected completions={len(all_completion_texts)} "
                    f"encoded_samples={len(encoded_samples)}"
                )
                if all_completion_texts:
                    _print_main(f"[post-rollout] completion_chars={len(all_completion_texts[0])}")
                    _print_main("[post-rollout] completion_full_begin")
                    _print_main(all_completion_texts[0])
                    _print_main("[post-rollout] completion_full_end")

            _validate_rollout_record_counts_collectively(
                encoded_count=len(encoded_samples),
                reward_input_count=len(local_reward_inputs),
                group_index_count=len(group_index),
                completion_text_count=len(all_completion_texts),
                expected_count=len(prompt_batch) * max(1, int(grpo_args.num_generations)),
            )

            if global_step == 0:
                _print_main(f"[post-rollout] computing rewards for {len(local_reward_inputs)} completions")
            reward_scores = _compute_distributed_reward_scores(
                reward_fn=reward_fn,
                local_reward_inputs=local_reward_inputs,
                reward_batch_fn=reward_batch_fn,
            )
            reward_scores = _validate_reward_score_array(
                reward_scores,
                expected_count=len(encoded_samples),
                required_components=4,
            )
            reward_variances_text = None
            if reward_scores.ndim == 2 and reward_scores.shape[1] >= 4:
                reward_variances = np.var(reward_scores[:, :4], axis=0, dtype=np.float64)
                reward_variances_text = np.array2string(
                    reward_variances,
                    precision=6,
                    separator=", ",
                    floatmode="fixed",
                )
            if global_step == 0:
                _print_main(f"[post-rollout] rewards ready count={len(reward_scores)}")
            if _is_main_process() and all_completion_texts and reward_scores.size > 0:
                preview_step = global_step + 1
                variance_suffix = (
                    f" reward_var={reward_variances_text}"
                    if reward_variances_text is not None
                    else ""
                )
                _print_main(
                    f"[sample step {preview_step}/{total_steps}] "
                    f"reward={reward_scores[0]} "
                    f"completion_chars={len(all_completion_texts[0])}"
                    f"{variance_suffix}"
                )

            old_log_probs = None
            ref_log_probs = None
            response_mask = None
            think_response_mask = None
            answer_response_mask = None
            ref_logprob_micro_batch_size = max(0, int(grpo_args.ref_logprob_micro_batch_size))
            policy_logprob_micro_batch_size = (
                ref_logprob_micro_batch_size
                if ref_logprob_micro_batch_size > 0
                else max(1, int(grpo_args.minibatch_size))
            )

            policy_was_training = model.training
            model.eval()
            with torch.no_grad():
                if global_step == 0:
                    _print_main(
                        "[post-rollout] computing old policy logprobs "
                        f"in micro-batches of {policy_logprob_micro_batch_size}"
                    )
                # (
                #     old_log_probs,
                #     response_mask,
                #     think_response_mask,
                #     answer_response_mask,
                # ) = _compute_logprobs_microbatched(
                #     model=model,
                #     encoded_samples=encoded_samples,
                #     pad_token_id=tokenizer.pad_token_id,
                #     device=device,
                #     temperature=grpo_args.temperature,
                #     micro_batch_size=policy_logprob_micro_batch_size,
                # )

                if ref_model is not None and ref_logprob_micro_batch_size > 0:
                    if global_step == 0:
                        _print_main(
                            "[post-rollout] computing reference logprobs "
                            f"in micro-batches of {ref_logprob_micro_batch_size}"
                        )
                    (
                        ref_log_probs,
                        response_mask,
                        think_response_mask,
                        answer_response_mask,
                    ) = _compute_logprobs_microbatched(
                        model=ref_model,
                        encoded_samples=encoded_samples,
                        pad_token_id=tokenizer.pad_token_id,
                        device=device,
                        temperature=grpo_args.temperature,
                        micro_batch_size=ref_logprob_micro_batch_size,
                    )
                else:
                    if global_step == 0:
                        _print_main(f"[post-rollout] collating rollout batch size={len(encoded_samples)}")
                    rollout_batch = _collate_mm_samples(encoded_samples, pad_token_id=tokenizer.pad_token_id)
                    rollout_batch = _to_device(rollout_batch, device)
                    if ref_model is not None:
                        if global_step == 0:
                            _print_main("[post-rollout] computing reference logprobs")
                        (
                            ref_log_probs,
                            response_mask,
                            think_response_mask,
                            answer_response_mask,
                        ) = _compute_completion_logprobs(
                            ref_model, rollout_batch, temperature=grpo_args.temperature
                        )
                    else:
                        (
                            response_mask,
                            think_response_mask,
                            answer_response_mask,
                        ) = _get_completion_masks_from_batch(rollout_batch)
                    del rollout_batch
            if policy_was_training:
                model.train()
            if global_step == 0:
                _print_main("[post-rollout] logprobs ready")

            # if (
            #     old_log_probs is None
            #     or response_mask is None
            #     or think_response_mask is None
            #     or answer_response_mask is None
            # ):
            #     raise RuntimeError("Old policy logprobs or completion masks were not computed for the rollout batch.")
            # _validate_finite_tensor_collectively("old_log_probs", old_log_probs)
            if ref_log_probs is not None:
                _validate_finite_tensor_collectively("ref_log_probs", ref_log_probs)

            response_mask_for_adv = (
                response_mask.to(device=device) if response_mask.device != device else response_mask
            )
            reward_scores_0 = torch.tensor(reward_scores[:, 0], dtype=torch.float32, device=device)
            reward_scores_1 = torch.tensor(reward_scores[:, 1], dtype=torch.float32, device=device)
            reward_scores_2 = torch.tensor(reward_scores[:, 2], dtype=torch.float32, device=device)
            reward_scores_3 = torch.tensor(reward_scores[:, 3], dtype=torch.float32, device=device)
            token_level_rewards_0 = _build_token_level_rewards(reward_scores_0, response_mask_for_adv)
            token_level_rewards_1 = _build_token_level_rewards(reward_scores_1, response_mask_for_adv)
            token_level_rewards_2 = _build_token_level_rewards(reward_scores_2, response_mask_for_adv)
            token_level_rewards_3 = _build_token_level_rewards(reward_scores_3, response_mask_for_adv)

            #token_level_rewards = _build_token_level_rewards(reward_scores_t, response_mask)
            # advantages, _ = core_algos.compute_grpo_outcome_advantage(
            #     token_level_rewards=token_level_rewards,
            #     response_mask=response_mask,
            #     index=np.array(group_index, dtype=object),
            #     epsilon=grpo_args.reward_eps,
            #     norm_adv_by_std_in_grpo=True,
            # )

            # These are sequence-level outcome rewards. Each component is
            # independently normalized within its prompt group, combined with
            # QWENVL_GRPO_REWARD_WEIGHTS, and broadcast to sampled action tokens.
            # The think/answer masks are intentionally not used for causal
            # segment credit assignment here.
            advantages_0, _ = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards_0,
                response_mask=response_mask_for_adv,
                index=np.array(group_index, dtype=object),
                epsilon=grpo_args.reward_eps,
                norm_adv_by_std_in_grpo=True,
            )

            advantages_1, _ = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards_1,
                response_mask=response_mask_for_adv,
                index=np.array(group_index, dtype=object),
                epsilon=grpo_args.reward_eps,
                norm_adv_by_std_in_grpo=True,
            )

            advantages_2, _ = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards_2,
                response_mask=response_mask_for_adv,
                index=np.array(group_index, dtype=object),
                epsilon=grpo_args.reward_eps,
                norm_adv_by_std_in_grpo=True,
            )

            advantages_3, _ = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards_3,
                response_mask=response_mask_for_adv,
                index=np.array(group_index, dtype=object),
                epsilon=grpo_args.reward_eps,
                norm_adv_by_std_in_grpo=True,
            )
            reward_weights = _reward_component_weights()
            advantages = (
                reward_weights[0] * advantages_0
                + reward_weights[1] * advantages_1
                + reward_weights[2] * advantages_2
                + reward_weights[3] * advantages_3
            )


            _validate_finite_tensor_collectively("advantages", advantages)


            if global_step == 0:
                _print_main("[post-rollout] advantages ready")

            train_records: List[Dict[str, Any]] = []
            for idx, sample in enumerate(encoded_samples):
                seq_len = sample["input_ids"].numel()
                log_len = seq_len - 1
                rec = dict(sample)
                # rec["old_log_probs"] = old_log_probs[idx, :log_len].detach().cpu() #
                rec["advantages"] = advantages[idx, :log_len].detach().cpu()
                rec["response_mask"] = response_mask[idx, :log_len].detach().cpu()
                rec["think_response_mask"] = think_response_mask[idx, :log_len].detach().cpu()
                rec["answer_response_mask"] = answer_response_mask[idx, :log_len].detach().cpu()
                if ref_log_probs is not None:
                    rec["ref_log_probs"] = ref_log_probs[idx, :log_len].detach().cpu()
                train_records.append(rec)
            if global_step == 0:
                _print_main(f"[post-rollout] train records ready count={len(train_records)}")

            _validate_rollout_minibatch_schedule(
                record_count=len(train_records),
                minibatch_size=int(grpo_args.minibatch_size),
                inner_epochs=int(grpo_args.inner_epochs),
                gradient_accumulation_steps=(
                    int(training_args.gradient_accumulation_steps)
                    if use_deepspeed
                    else 1
                ),
            )

            step_loss = 0.0
            step_kl = 0.0
            step_pg = 0.0
            mb_count = 0
            sample_indices = list(range(len(train_records)))
            for _ in range(max(1, grpo_args.inner_epochs)):
                random.shuffle(sample_indices)
                for start in range(0, len(sample_indices), grpo_args.minibatch_size):
                    mb_ids = sample_indices[start : start + grpo_args.minibatch_size]
                    mb_records = [train_records[i] for i in mb_ids]
                    mbatch = _collate_train_records(mb_records, pad_token_id=tokenizer.pad_token_id)
                    mbatch = _to_device(mbatch, device)

                    current_log_probs, _, _, _ = _compute_completion_logprobs(
                        model, mbatch, temperature=grpo_args.temperature
                    )
                    _validate_finite_tensor_collectively("current_log_probs", current_log_probs)
                    if global_step == 0 and mb_count == 0:
                        _print_main(
                            f"[train] first minibatch ready size={len(mb_records)} "
                            f"seq_shape={tuple(mbatch['input_ids'].shape)}"
                        )
                    pg_loss, _, _, _ = core_algos.compute_policy_loss(
                        old_log_prob=current_log_probs.detach(),  #old_log_prob=current_log_probs.detach(),mbatch["old_log_probs"]
                        log_prob=current_log_probs,
                        advantages=mbatch["advantages"],
                        response_mask=mbatch["response_mask"],
                        cliprange=grpo_args.clip_eps,
                        cliprange_low=grpo_args.clip_eps,
                        cliprange_high=grpo_args.clip_eps,
                        loss_agg_mode="token-mean",
                    )

                    kl_loss = torch.zeros((), device=device)
                    if "ref_log_probs" in mbatch:
                        kl_token = core_algos.kl_penalty(
                            current_log_probs, mbatch["ref_log_probs"], kl_penalty="low_var_kl"
                        )
                        denom = mbatch["response_mask"].sum().clamp_min(1.0)
                        kl_loss = (kl_token * mbatch["response_mask"]).sum() / denom

                    loss = pg_loss + grpo_args.beta_kl * kl_loss
                    _validate_finite_tensor_collectively(
                        "minibatch_losses",
                        torch.stack((pg_loss, kl_loss, loss)),
                    )
                    if use_deepspeed:
                        model.backward(loss)
                        model.step()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable_params, training_args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    step_loss += float(loss.item())
                    step_kl += float(kl_loss.item())
                    step_pg += float(pg_loss.item())
                    mb_count += 1
                    del current_log_probs, pg_loss, kl_loss, loss, mbatch, mb_records
                    if "kl_token" in locals():
                        del kl_token

            train_record_count = len(train_records)
            if reward_scores.size > 0:
                avg_reward = float(np.mean(reward_scores[:, :4]))
            else:
                avg_reward = 0.0
            del (
                responses,
                response_map,
                encoded_samples,
                local_reward_inputs,
                group_index,
                all_completion_texts,
                ref_log_probs,
                response_mask,
                think_response_mask,
                answer_response_mask,
                response_mask_for_adv,
                reward_scores_0,
                reward_scores_1,
                reward_scores_2,
                reward_scores_3,
                token_level_rewards_0,
                token_level_rewards_1,
                token_level_rewards_2,
                token_level_rewards_3,
                advantages_0,
                advantages_1,
                advantages_2,
                advantages_3,
                advantages,
                train_records,
            )
            _release_cuda_cache()
            global_step += 1
            next_resume_epoch = epoch
            next_resume_batch_idx = batch_idx + 1
            next_resume_global_sample_cursor = (
                next_resume_batch_idx * world_size * per_device_batch_size
            )
            if next_resume_batch_idx >= num_epoch_batches:
                next_resume_epoch = epoch + 1
                next_resume_batch_idx = 0
                next_resume_global_sample_cursor = 0
            if hasattr(rollout_backend, "maybe_sync_from_policy_model"):
                rollout_backend.maybe_sync_from_policy_model(model, global_step)
            progress_pct = 100.0 * float(global_step) / float(max(1, total_steps))
            if _is_main_process():
                _print_main(
                    f"[progress] step={global_step}/{total_steps} "
                    f"percent={progress_pct:.1f}% avg_reward={avg_reward:.6f} "
                    f"samples={train_record_count}"
                )
            if global_step % training_args.logging_steps == 0:
                global_step_loss = _reduce_scalar_sum(step_loss)
                global_step_kl = _reduce_scalar_sum(step_kl)
                global_step_pg = _reduce_scalar_sum(step_pg)
                global_mb_count = max(1.0, _reduce_scalar_sum(float(mb_count)))
                global_avg_reward = _reduce_scalar_mean(avg_reward)
                global_samples = int(round(_reduce_scalar_sum(float(train_record_count))))
                if _is_main_process():
                    _print_main(
                        f"[step {global_step}] loss={global_step_loss / global_mb_count:.6f} "
                        f"pg={global_step_pg / global_mb_count:.6f} kl={global_step_kl / global_mb_count:.6f} "
                        f"avg_reward={global_avg_reward:.6f} samples={global_samples}"
                    )
                    tb_writer = _write_tensorboard_scalars(
                        tb_writer,
                        step=global_step,
                        loss=global_step_loss / global_mb_count,
                        pg=global_step_pg / global_mb_count,
                        kl=global_step_kl / global_mb_count,
                        avg_reward=global_avg_reward,
                    )

            save_every = bridge_args.save_every_steps if bridge_args.save_every_steps > 0 else training_args.save_steps
            if save_every > 0 and global_step % save_every == 0:
                _save_checkpoint(
                    model,
                    processor,
                    training_args.output_dir,
                    global_step,
                    epoch=next_resume_epoch,
                    batch_idx=next_resume_batch_idx,
                    dataset_size=len(prompt_dataset),
                    batch_size=per_device_batch_size,
                    seed=training_args.seed,
                    dataset_use=data_args.dataset_use,
                    max_prompt_samples=bridge_args.max_prompt_samples,
                    rollout_num_workers=bridge_args.rollout_num_workers,
                    enable_local_image=getattr(data_args, "enable_local_image", False),
                    global_sample_cursor=next_resume_global_sample_cursor,
                    data_reference_world_size=data_reference_world_size,
                    data_reference_batch_size=data_reference_batch_size,
                    warm_start_source_checkpoint=warm_start_checkpoint,
                )
                _rotate_checkpoints(
                    training_args.output_dir,
                    getattr(training_args, "save_total_limit", None),
                )

        epoch += 1

    if bridge_args.skip_final_save:
        _barrier()
        if tb_writer is not None:
            try:
                tb_writer.close()
            except Exception:
                pass
        _print_main(
            "[smoke-test] training loop completed successfully; final checkpoint save was skipped"
        )
        return

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.config.use_cache = True
    final_client_state = {
        "global_step": int(global_step),
        "epoch": int(next_resume_epoch),
        "batch_idx": int(next_resume_batch_idx),
        "global_sample_cursor": int(next_resume_global_sample_cursor),
        "data_reference_world_size": int(data_reference_world_size),
        "data_reference_batch_size": int(data_reference_batch_size),
    }
    if use_deepspeed:
        _barrier()
        if _is_peft_model(model_to_save):
            if _is_main_process():
                _save_hf_artifacts(model_to_save, processor, training_args.output_dir)
                model_to_save.save_pretrained(training_args.output_dir)
        else:
            if _is_main_process():
                _save_hf_artifacts(model_to_save, processor, training_args.output_dir)
            if hasattr(model, "save_16bit_model"):
                model.save_16bit_model(training_args.output_dir, save_filename="pytorch_model.bin")
            model.save_checkpoint(
                training_args.output_dir,
                tag=DEEPSPEED_CHECKPOINT_TAG,
                client_state=final_client_state,
            )
        _save_trainer_state(
            training_args.output_dir,
            global_step=global_step,
            epoch=next_resume_epoch,
            batch_idx=next_resume_batch_idx,
            dataset_size=len(prompt_dataset),
            batch_size=per_device_batch_size,
            seed=training_args.seed,
            dataset_use=data_args.dataset_use,
            max_prompt_samples=bridge_args.max_prompt_samples,
            rollout_num_workers=bridge_args.rollout_num_workers,
            enable_local_image=getattr(data_args, "enable_local_image", False),
            global_sample_cursor=next_resume_global_sample_cursor,
            data_reference_world_size=data_reference_world_size,
            data_reference_batch_size=data_reference_batch_size,
            warm_start_source_checkpoint=warm_start_checkpoint,
        )
        _barrier()
    else:
        model_to_save.save_pretrained(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)
        _save_trainer_state(
            training_args.output_dir,
            global_step=global_step,
            epoch=next_resume_epoch,
            batch_idx=next_resume_batch_idx,
            dataset_size=len(prompt_dataset),
            batch_size=per_device_batch_size,
            seed=training_args.seed,
            dataset_use=data_args.dataset_use,
            max_prompt_samples=bridge_args.max_prompt_samples,
            rollout_num_workers=bridge_args.rollout_num_workers,
            enable_local_image=getattr(data_args, "enable_local_image", False),
            global_sample_cursor=next_resume_global_sample_cursor,
            data_reference_world_size=data_reference_world_size,
            data_reference_batch_size=data_reference_batch_size,
            warm_start_source_checkpoint=warm_start_checkpoint,
        )

    if tb_writer is not None:
        try:
            tb_writer.close()
        except Exception:
            pass

    _print_main(f"Training finished. Saved to: {training_args.output_dir}")


if __name__ == "__main__":
    train()
