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

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path
import copy
from types import MethodType

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.train.trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

import torch
from torch import nn
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig




def _parameter_logical_shape(parameter):
    """Return the unpartitioned shape for ordinary and ZeRO-3 parameters."""

    return tuple(getattr(parameter, "ds_shape", parameter.shape))


def _copy_module_state(source_module, target_module):
    """Copy module state without treating ZeRO-3 placeholders as real weights.

    Under ZeRO-3, ``state_dict()`` exposes partitioned parameters as zero-sized
    placeholders outside a gather context. A normal ``load_state_dict`` can
    therefore either fail with a size mismatch or silently copy no data. Both
    modules are gathered here and rank 0 performs the actual copy; DeepSpeed
    broadcasts and repartitions the modified target parameters on context exit.
    """

    source_parameters = dict(source_module.named_parameters())
    target_parameters = dict(target_module.named_parameters())
    if source_parameters.keys() != target_parameters.keys():
        missing = sorted(source_parameters.keys() - target_parameters.keys())
        unexpected = sorted(target_parameters.keys() - source_parameters.keys())
        raise ValueError(
            "Cannot clone visual branch with different parameter names: "
            f"missing={missing}, unexpected={unexpected}."
        )

    parameter_pairs = [
        (name, source_parameters[name], target_parameters[name])
        for name in source_parameters
    ]
    for name, source_parameter, target_parameter in parameter_pairs:
        source_shape = _parameter_logical_shape(source_parameter)
        target_shape = _parameter_logical_shape(target_parameter)
        if source_shape != target_shape:
            raise ValueError(
                f"Cannot clone visual parameter {name!r}: "
                f"source shape {source_shape} != target shape {target_shape}."
            )

    zero_patterns = {
        (hasattr(source, "ds_id"), hasattr(target, "ds_id"))
        for _, source, target in parameter_pairs
    }

    def copy_pairs(pairs, *, rank_zero_only):
        should_copy = (
            not rank_zero_only
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        if should_copy:
            with torch.no_grad():
                for _, source_parameter, target_parameter in pairs:
                    target_parameter.copy_(source_parameter)

    if not parameter_pairs or zero_patterns == {(False, False)}:
        target_module.load_state_dict(source_module.state_dict(), strict=True)
        return

    try:
        import deepspeed
    except ImportError as exc:  # pragma: no cover - inconsistent runtime setup
        raise RuntimeError(
            "Found ZeRO-partitioned visual parameters but DeepSpeed is unavailable."
        ) from exc

    if zero_patterns == {(True, True)}:
        zero_parameters = [
            parameter
            for _, source, target in parameter_pairs
            for parameter in (source, target)
        ]
        with deepspeed.zero.GatheredParameters(
            zero_parameters, modifier_rank=0
        ):
            copy_pairs(parameter_pairs, rank_zero_only=True)
    elif zero_patterns == {(True, False)}:
        with deepspeed.zero.GatheredParameters(
            [source for _, source, _ in parameter_pairs], modifier_rank=None
        ):
            copy_pairs(parameter_pairs, rank_zero_only=False)
    elif zero_patterns == {(False, True)}:
        with deepspeed.zero.GatheredParameters(
            [target for _, _, target in parameter_pairs], modifier_rank=0
        ):
            copy_pairs(parameter_pairs, rank_zero_only=True)
    else:
        # Future architectures may mix partitioned and ordinary parameters.
        for pair in parameter_pairs:
            _, source_parameter, target_parameter = pair
            source_is_zero = hasattr(source_parameter, "ds_id")
            target_is_zero = hasattr(target_parameter, "ds_id")
            if source_is_zero and target_is_zero:
                with deepspeed.zero.GatheredParameters(
                    [source_parameter, target_parameter], modifier_rank=0
                ):
                    copy_pairs([pair], rank_zero_only=True)
            elif source_is_zero:
                with deepspeed.zero.GatheredParameters(
                    [source_parameter], modifier_rank=None
                ):
                    copy_pairs([pair], rank_zero_only=False)
            elif target_is_zero:
                with deepspeed.zero.GatheredParameters(
                    [target_parameter], modifier_rank=0
                ):
                    copy_pairs([pair], rank_zero_only=True)
            else:
                copy_pairs([pair], rank_zero_only=False)

    source_buffers = dict(source_module.named_buffers())
    target_buffers = dict(target_module.named_buffers())
    if source_buffers.keys() != target_buffers.keys():
        missing = sorted(source_buffers.keys() - target_buffers.keys())
        unexpected = sorted(target_buffers.keys() - source_buffers.keys())
        raise ValueError(
            "Cannot clone visual branch with different buffer names: "
            f"missing={missing}, unexpected={unexpected}."
        )
    with torch.no_grad():
        for name, source_buffer in source_buffers.items():
            target_buffer = target_buffers[name]
            if source_buffer.shape != target_buffer.shape:
                raise ValueError(
                    f"Cannot clone visual buffer {name!r}: "
                    f"source shape {tuple(source_buffer.shape)} != "
                    f"target shape {tuple(target_buffer.shape)}."
                )
            target_buffer.copy_(source_buffer)


def attach_local_vision_branch(model):
    """Attach a duplicated visual branch used for local-image encoding."""
    if not hasattr(model, "model") or not hasattr(model.model, "visual"):
        raise ValueError("Current model does not expose model.visual; cannot attach local visual branch.")
    model.config.use_local_vision_branch = True
    core_model = model.model
    if hasattr(core_model, "local_visual"):
        return model

    # Build a fresh vision module from config, then copy the real (possibly
    # partitioned) weights rather than zero-sized ZeRO-3 placeholders.
    local_visual = type(core_model.visual)._from_config(core_model.config.vision_config)
    _copy_module_state(core_model.visual, local_visual)
    local_visual.train(core_model.visual.training)
    core_model.local_visual = local_visual
    core_model._origin_forward = core_model.forward
    core_model._origin_get_image_features = core_model.get_image_features
    core_model._image_branch_flags = None

    def forward_with_local(self, *args, **kwargs):
        self._image_branch_flags = kwargs.pop("image_branch_flags", None)
        try:
            return self._origin_forward(*args, **kwargs)
        finally:
            self._image_branch_flags = None

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
                # The zero-valued dependency is needed only while gradients
                # and distributed unused-parameter accounting are active. In
                # rollout it would run a complete unused vision tower.
                if not torch.is_grad_enabled():
                    return {}, {}, None
                # Keep parameter usage consistent across ranks for ZeRO/DDP:
                # build a zero-valued dependency on this branch even when no sample hits it.
                fallback_pixels = pixel_splits[0].type(visual_module.dtype)
                fallback_grid_thw = image_grid_thw[0:1]
                dummy_embeds, dummy_deepstack = visual_module(fallback_pixels, grid_thw=fallback_grid_thw)
                dummy_term = dummy_embeds.sum() * 0.0
                for layer_tensor in dummy_deepstack:
                    dummy_term = dummy_term + layer_tensor.sum() * 0.0
                return {}, {}, dummy_term
            grouped_pixels = torch.cat([pixel_splits[i] for i in sample_ids], dim=0).type(visual_module.dtype)
            grouped_grid_thw = torch.stack([image_grid_thw[i] for i in sample_ids], dim=0)
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
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if data_args.enable_local_image:
        model_args.use_local_vision_branch = True
    if model_args.use_local_vision_branch:
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

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        # Default LoRA targets for LLM attention
        lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        visual_lora_targets = []
        for module_name, module in model.named_modules():
            if "visual" in module_name and isinstance(module, torch.nn.Linear):
                visual_lora_targets.append(module_name)
        if len(visual_lora_targets) > 0:
            lora_targets.extend(visual_lora_targets)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    print(f"Added {len(visual_lora_targets)} visual LoRA target modules")
            else:
                print(f"Added {len(visual_lora_targets)} visual LoRA target modules")

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
            target_modules=lora_targets,  # LLM + visual + local_visual linear layers
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
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
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
