# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import contextlib
import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path
from types import MethodType
from typing import Optional
from dataclasses import dataclass, field
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.train.trainer import replace_qwen2_vl_attention_class  # noqa: E402
from qwenvl.train.checkpoint_loader import load_checkpoint_into_model  # noqa: E402

from transformers import (  # noqa: E402
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module  # noqa: E402
from qwenvl.train.argument import (  # noqa: E402
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer  # noqa: E402



def attach_local_vision_branch(model):
    """Attach a duplicated visual branch used for local-image encoding."""
    if not hasattr(model, "model") or not hasattr(model.model, "visual"):
        raise ValueError("Current model does not expose model.visual; cannot attach local visual branch.")
    model.config.use_local_vision_branch = True
    core_model = model.model

    existing_local_visual = getattr(core_model, "local_visual", None)
    has_forward_patch = hasattr(core_model, "_origin_forward") and hasattr(core_model, "_origin_get_image_features")
    has_initialized_local_visual = False
    if existing_local_visual is not None:
        try:
            has_initialized_local_visual = any(p.numel() > 0 for p in existing_local_visual.parameters())
        except Exception:
            has_initialized_local_visual = False

    if has_initialized_local_visual and has_forward_patch:
        return model

    if not has_initialized_local_visual:
        # Build a fresh vision module from config, then copy weights.
        # This is more stable than deepcopy under DeepSpeed ZeRO-3.
        local_visual = type(core_model.visual)._from_config(core_model.config.vision_config)
        local_visual.load_state_dict(core_model.visual.state_dict(), strict=True)
        core_model.local_visual = local_visual
    # ZeRO-3 compatibility: local_visual is consumed inside patched core forward path.
    # Registering its params as external avoids NOT_AVAILABLE assertion in some DS setups.
    try:
        import deepspeed

        for p in core_model.local_visual.parameters():
            deepspeed.zero.register_external_parameter(core_model, p)
    except Exception:
        pass
    core_model._origin_forward = core_model.forward
    core_model._origin_get_image_features = core_model.get_image_features
    core_model._image_branch_flags = None

    def forward_with_local(self, *args, **kwargs):
        self._image_branch_flags = kwargs.pop("image_branch_flags", None)
        try:
            return self._origin_forward(*args, **kwargs)
        finally:
            self._image_branch_flags = None

    def _maybe_gather_zero3_module_params(module):
        def _is_partitioned_param(param: torch.nn.Parameter) -> bool:
            ds_id = getattr(param, "ds_id", None)
            ds_shape = getattr(param, "ds_shape", None)
            if ds_id is None and ds_shape is None:
                return False
            if ds_id is not None and ds_shape is None:
                return True
            if ds_shape is None:
                return False
            full_shape = tuple(int(dim) for dim in ds_shape)
            local_shape = tuple(int(dim) for dim in param.shape)
            if local_shape != full_shape:
                return True
            expected_numel = 1
            for dim in full_shape:
                expected_numel *= max(1, int(dim))
            return int(param.numel()) != expected_numel

        params = [p for p in module.parameters() if _is_partitioned_param(p)]
        if not params:
            return contextlib.nullcontext()
        try:
            import deepspeed

            return deepspeed.zero.GatheredParameters(params, modifier_rank=None)
        except Exception:
            return contextlib.nullcontext()

    def get_image_features_with_local(self, pixel_values: torch.FloatTensor, image_grid_thw=None):
        image_branch_flags = self._image_branch_flags
        if image_branch_flags is None or image_grid_thw is None:
            return self._origin_get_image_features(pixel_values, image_grid_thw)

        if isinstance(image_branch_flags, torch.Tensor):
            image_branch_flags = image_branch_flags.tolist()
        if len(image_branch_flags) != image_grid_thw.shape[0]:
            return self._origin_get_image_features(pixel_values, image_grid_thw)
        if len(image_branch_flags) == 0:
            return self._origin_get_image_features(pixel_values, image_grid_thw)

        raw_split_sizes = image_grid_thw.prod(-1).tolist()
        pixel_splits = list(torch.split(pixel_values, raw_split_sizes))
        global_ids = [i for i, f in enumerate(image_branch_flags) if int(f) == 0]
        local_ids = [i for i, f in enumerate(image_branch_flags) if int(f) == 1]

        def _encode_group(sample_ids, visual_module):
            if len(sample_ids) == 0:
                # Skip the otherwise full dummy vision-tower forward during
                # no-grad rollout/inference. Training keeps the dependency
                # required by ZeRO/DDP unused-parameter accounting.
                if not torch.is_grad_enabled():
                    return {}, {}, None
                # Keep parameter usage consistent across ranks for ZeRO/DDP:
                # build a zero-valued dependency on this branch even when no sample hits it.
                fallback_pixels = pixel_splits[0].type(visual_module.dtype)
                fallback_grid_thw = image_grid_thw[0:1]
                with _maybe_gather_zero3_module_params(visual_module):
                    dummy_embeds, dummy_deepstack = visual_module(fallback_pixels, grid_thw=fallback_grid_thw)
                dummy_term = dummy_embeds.sum() * 0.0
                for layer_tensor in dummy_deepstack:
                    dummy_term = dummy_term + layer_tensor.sum() * 0.0
                return {}, {}, dummy_term
            grouped_pixels = torch.cat([pixel_splits[i] for i in sample_ids], dim=0).type(visual_module.dtype)
            grouped_grid_thw = torch.stack([image_grid_thw[i] for i in sample_ids], dim=0)
            with _maybe_gather_zero3_module_params(visual_module):
                embeds_cat, deepstack_cat = visual_module(grouped_pixels, grid_thw=grouped_grid_thw)
            split_sizes = (grouped_grid_thw.prod(-1) // visual_module.spatial_merge_size**2).tolist()
            embeds_split = torch.split(embeds_cat, split_sizes)
            embed_map = {sid: emb for sid, emb in zip(sample_ids, embeds_split)}
            
            deepstack_map = {}
            for layer_idx, layer_cat in enumerate(deepstack_cat):
                layer_split = torch.split(layer_cat, split_sizes)
                deepstack_map[layer_idx] = {sid: emb for sid, emb in zip(sample_ids, layer_split)}
            return embed_map, deepstack_map, None

        global_embed_map, global_deep_map, global_dummy_term = _encode_group(global_ids, self.visual)
        local_embed_map, local_deep_map, local_dummy_term = _encode_group(local_ids, self.local_visual)
        dummy_term = None
        if global_dummy_term is not None:
            dummy_term = global_dummy_term
        if local_dummy_term is not None:
            dummy_term = local_dummy_term if dummy_term is None else (dummy_term + local_dummy_term)

        image_embed_splits = []
        for idx in range(len(image_branch_flags)):
            if idx in global_embed_map:
                selected_embeds = global_embed_map[idx]
            else:
                selected_embeds = local_embed_map[idx]
            if dummy_term is not None and idx == 0:
                selected_embeds = selected_embeds + dummy_term
            image_embed_splits.append(selected_embeds)

        num_deep_layers = max(len(global_deep_map), len(local_deep_map))
        deepstack_image_embeds = []
        for layer_idx in range(num_deep_layers):
            layer_parts = []
            global_layer_map = global_deep_map.get(layer_idx, {})
            local_layer_map = local_deep_map.get(layer_idx, {})
            for idx in range(len(image_branch_flags)):
                if idx in global_layer_map:
                    layer_parts.append(global_layer_map[idx])
                else:
                    layer_parts.append(local_layer_map[idx])
            deepstack_image_embeds.append(torch.cat(layer_parts, dim=0))

        self._image_branch_flags = None
        return tuple(image_embed_splits), deepstack_image_embeds

    core_model.forward = MethodType(forward_with_local, core_model)
    core_model.get_image_features = MethodType(get_image_features_with_local, core_model)
    return model


def get_local_visual_module(model):
    if hasattr(model, "model") and hasattr(model.model, "local_visual"):
        return model.model.local_visual
    if hasattr(model, "base_model"):
        base_model = model.base_model
        if hasattr(base_model, "model") and hasattr(base_model.model, "model"):
            core_model = base_model.model.model
            if hasattr(core_model, "local_visual"):
                return core_model.local_visual
    return None


@dataclass
class LocalBranchArguments:
    local_branch_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to local branch checkpoint source. Defaults to model_name_or_path."},
    )
    local_branch_strict_load: bool = field(
        default=True,
        metadata={"help": "Raise error when local branch weights cannot be loaded into model.local_visual."},
    )


@dataclass
class LossWeightArguments:
    cot_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Loss weight for CoT tokens inside <think>...</think>."},
    )
    answer_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Loss weight for answer tokens after </think>."},
    )
    think_start_text: str = field(
        default="<think>",
        metadata={"help": "CoT start marker text."},
    )
    think_end_text: str = field(
        default="</think>",
        metadata={"help": "CoT end marker text."},
    )


def _find_subseq(seq, pattern, start=0):
    if len(pattern) == 0:
        return -1
    max_i = len(seq) - len(pattern) + 1
    for i in range(start, max_i):
        if seq[i : i + len(pattern)] == pattern:
            return i
    return -1


class WeightedCotAnswerTrainer(Trainer):
    def __init__(
        self,
        *args,
        cot_loss_weight: float = 1.0,
        answer_loss_weight: float = 1.0,
        think_start_ids=None,
        think_end_ids=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cot_loss_weight = float(cot_loss_weight)
        self.answer_loss_weight = float(answer_loss_weight)
        self.think_start_ids = think_start_ids or []
        self.think_end_ids = think_end_ids or []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_input_ids = inputs["input_ids"][..., 1:].contiguous()

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        valid_mask = shift_labels.ne(-100)
        weight = torch.ones_like(loss_per_token, dtype=loss_per_token.dtype)
        cot_mask = torch.zeros_like(valid_mask)
        ans_mask = torch.zeros_like(valid_mask)

        if len(self.think_start_ids) > 0 and len(self.think_end_ids) > 0:
            bsz = shift_input_ids.size(0)
            seqlen = shift_input_ids.size(1)
            for b in range(bsz):
                seq = shift_input_ids[b].tolist()
                start_tag = _find_subseq(seq, self.think_start_ids, start=0)
                if start_tag < 0:
                    ans_mask[b, :] = True
                    continue
                cot_start = start_tag + len(self.think_start_ids)
                end_tag = _find_subseq(seq, self.think_end_ids, start=cot_start)
                if end_tag < 0:
                    ans_mask[b, :] = True
                    continue
                if cot_start < end_tag:
                    cot_mask[b, cot_start:end_tag] = True
                ans_start = end_tag + len(self.think_end_ids)
                if ans_start < seqlen:
                    ans_mask[b, ans_start:seqlen] = True
        else:
            ans_mask[:] = True

        other_mask = valid_mask & (~cot_mask) & (~ans_mask)
        weight = torch.where(cot_mask, torch.full_like(weight, self.cot_loss_weight), weight)
        weight = torch.where(ans_mask, torch.full_like(weight, self.answer_loss_weight), weight)
        weight = torch.where(other_mask, torch.ones_like(weight), weight)

        weighted_valid = weight * valid_mask.to(weight.dtype)
        loss = (loss_per_token * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)

        return (loss, outputs) if return_outputs else loss


def _is_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def load_local_branch_weights_from_model_path(model, model_path: str, strict_on_no_match: bool = False):
    """Overlay a full checkpoint and require exact local weights in strict mode.

    The historical function name is retained for launcher compatibility.  The
    checkpoint is a full SFT/GRPO policy overlay, so compatible decoder and
    global-vision tensors are restored as well.  Loading is streamed by shard
    by :mod:`qwenvl.train.checkpoint_loader`.
    """

    if not model_path:
        if strict_on_no_match:
            raise ValueError(
                "[CheckpointLoad] strict local-vision loading requires a checkpoint path."
            )
        return
    if not Path(model_path).exists():
        msg = f"[CheckpointLoad] model_path={model_path} does not exist on local fs."
        if strict_on_no_match:
            raise FileNotFoundError(msg)
        if _is_main_process():
            print(f"{msg} Skip reload.")
        return
    return load_checkpoint_into_model(
        model,
        model_path,
        scope="all",
        strict=bool(strict_on_no_match),
        require_local=bool(strict_on_no_match),
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

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
            p.requires_grad = tune_local_vision
        for _, p in model.model.local_visual.merger.named_parameters():
            p.requires_grad = tune_local_mlp



def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LossWeightArguments)
    )

    model_args, data_args, training_args, loss_weight_args = parser.parse_args_into_dataclasses()
    # This entry is dedicated to local-branch training.
    model_args.use_local_vision_branch = True
    data_args.enable_local_image = True
   
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        if model_args.use_local_vision_branch:
            model = attach_local_vision_branch(model)
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        if model_args.use_local_vision_branch:
            model = attach_local_vision_branch(model)
      
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    if not hasattr(model, "model") or not hasattr(model.model, "local_visual"):
        raise ValueError("Local vision branch is not attached.")
    local_branch_path = model_args.local_branch_model_path or model_args.model_name_or_path
    load_local_branch_weights_from_model_path(
        model,
        local_branch_path,
        strict_on_no_match=model_args.local_branch_strict_load,
    )

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    think_start_ids = tokenizer.encode(loss_weight_args.think_start_text, add_special_tokens=False)
    think_end_ids = tokenizer.encode(loss_weight_args.think_end_text, add_special_tokens=False)
    if _is_main_process():
        print(
            f"[LossWeight] cot={loss_weight_args.cot_loss_weight}, "
            f"answer={loss_weight_args.answer_loss_weight}, "
            f"think_start_ids={think_start_ids}, think_end_ids={think_end_ids}"
        )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        # Default LoRA targets for LLM attention
        lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        # Add all Linear layers under local_visual for true LoRA tuning on local branch
        if model_args.use_local_vision_branch:
            local_lora_targets = []
            for module_name, module in model.named_modules():
                if "local_visual" in module_name and isinstance(module, torch.nn.Linear):
                    local_lora_targets.append(module_name)
            if len(local_lora_targets) > 0:
                lora_targets.extend(local_lora_targets)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        print(f"Added {len(local_lora_targets)} local_visual LoRA target modules")
                else:
                    print(f"Added {len(local_lora_targets)} local_visual LoRA target modules")

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=lora_targets,  # LLM + local_visual linear layers
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
            if hasattr(model.model, "local_visual"):
                print("Local visual branch trainable status:")
                model.model.local_visual.print_trainable_parameters()

    
    data_module = make_supervised_data_module(
        processor, data_args=data_args, sampling_seed=training_args.seed
    )
    trainer = WeightedCotAnswerTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        cot_loss_weight=loss_weight_args.cot_loss_weight,
        answer_loss_weight=loss_weight_args.answer_loss_weight,
        think_start_ids=think_start_ids,
        think_end_ids=think_end_ids,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
