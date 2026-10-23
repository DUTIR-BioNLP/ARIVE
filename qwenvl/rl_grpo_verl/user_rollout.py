from __future__ import annotations

import contextlib
import copy
import os
import re
import traceback
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwenvl.data.data_processor import load_rgb_bbox, parse_path_and_bbox

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams
from vllm.multimodal.inputs import MultiModalFieldConfig

from .interfaces import Completion, RolloutRequest, RolloutResponse
from .rollout_backend import BaseRolloutBackend


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _dist_is_initialized() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _dist_is_initialized() else 1


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _is_out_of_memory_error(exc: BaseException) -> bool:
    """Do not downgrade allocator failures to a recoverable bad sample."""
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


@contextlib.contextmanager
def _temporarily_scrub_torchrun_env():
    scrub_keys = [
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ]
    scrub_prefixes = ("TORCHELASTIC_",)
    backup = {}
    removed = []
    for key in list(os.environ):
        if key in scrub_keys or any(key.startswith(prefix) for prefix in scrub_prefixes):
            backup[key] = os.environ[key]
            removed.append(key)
            os.environ.pop(key, None)
    try:
        yield removed
    finally:
        for key, value in backup.items():
            os.environ[key] = value


@contextlib.contextmanager
def _temporarily_set_env(name: str, value: Optional[str]):
    had_original = name in os.environ
    original = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if had_original:
            os.environ[name] = original if original is not None else ""
        else:
            os.environ.pop(name, None)


@contextlib.contextmanager
def _temporarily_disable_deepspeed_zero_init():
    try:
        import deepspeed
    except Exception:
        yield
        return

    partition_parameters = getattr(getattr(deepspeed, "zero", None), "partition_parameters", None)
    if partition_parameters is None:
        yield
        return

    shutdown_init_context = getattr(partition_parameters, "shutdown_init_context", None)
    restore_init_context = getattr(partition_parameters, "restore_init_context", None)
    if shutdown_init_context is None or restore_init_context is None:
        yield
        return

    shutdown_init_context()
    try:
        yield
    finally:
        restore_init_context()


class _RolloutVisionCore(torch.nn.Module):
    def __init__(self, visual: torch.nn.Module, local_visual: Optional[torch.nn.Module] = None):
        super().__init__()
        self.visual = visual
        if local_visual is not None:
            self.local_visual = local_visual


class _RolloutVisionModel(torch.nn.Module):
    def __init__(self, core_model: _RolloutVisionCore):
        super().__init__()
        self.model = core_model


def check_vllm_mm_embed_capability():
    init_vars = LLM.__init__.__code__.co_varnames
    has_enable_mm_embeds = "enable_mm_embeds" in init_vars
    print(
        f"[UserRolloutBackend] vLLM enable_mm_embeds available: {has_enable_mm_embeds}. "
        "Using multi_modal_data external embeds path.",
        flush=True,
    )


def patch_vllm_qwen3vl_mm_embed_field_sizes():
    try:
        from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
        from vllm.multimodal.parse import DictEmbeddingItems, MultiModalDataParser
    except Exception as exc:
        print(f"[UserRolloutBackend] Skip vLLM runtime patch: {exc}")
        return

    def _create_qwen3vl_field_factory(spatial_merge_size: int):
        def _qwen3vl_field_config(hf_inputs):
            image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
            image_grid_sizes = image_grid_thw.prod(-1)
            image_embed_sizes = image_grid_sizes // spatial_merge_size // spatial_merge_size

            video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
            video_grid_sizes = video_grid_thw.prod(-1)
            video_embed_sizes = video_grid_sizes // spatial_merge_size // spatial_merge_size

            return dict(
                pixel_values=MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes),
                image_embeds=MultiModalFieldConfig.flat_from_sizes("image", image_embed_sizes),
                image_grid_thw=MultiModalFieldConfig.batched("image"),
                pixel_values_videos=MultiModalFieldConfig.flat_from_sizes("video", video_grid_sizes),
                video_embeds=MultiModalFieldConfig.flat_from_sizes("video", video_embed_sizes),
                video_grid_thw=MultiModalFieldConfig.batched("video"),
            )

        return _qwen3vl_field_config

    class _PatchedQwen3VLDataParser(MultiModalDataParser):
        def __init__(self, spatial_merge_size: int, *args, **kwargs):
            self._spatial_merge_size = spatial_merge_size
            super().__init__(*args, **kwargs)

        def _parse_image_data(self, data):
            if isinstance(data, dict):
                return DictEmbeddingItems(
                    data,
                    modality="image",
                    required_fields={"image_embeds", "image_grid_thw"},
                    fields_factory=_create_qwen3vl_field_factory(self._spatial_merge_size),
                )
            return super()._parse_image_data(data)

        def _parse_video_data(self, data):
            if isinstance(data, dict):
                return DictEmbeddingItems(
                    data,
                    modality="video",
                    required_fields={"video_embeds", "video_grid_thw"},
                    fields_factory=_create_qwen3vl_field_factory(self._spatial_merge_size),
                )
            return super()._parse_video_data(data)

    def _patched_get_data_parser(self):
        image_processor = self.info.get_image_processor()
        merge_size = int(image_processor.merge_size)
        return _PatchedQwen3VLDataParser(
            spatial_merge_size=merge_size,
            video_needs_metadata=True,
        )

    def _patched_get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_grid_sizes = image_grid_thw.prod(-1)

        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        video_grid_sizes = video_grid_thw.prod(-1)

        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        merge_size = int(image_processor.merge_size)
        merge_area = merge_size * merge_size

        image_embed_sizes = image_grid_sizes // merge_area
        video_embed_sizes = video_grid_sizes // merge_area

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes),
            image_embeds=MultiModalFieldConfig.flat_from_sizes("image", image_embed_sizes),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes("video", video_grid_sizes),
            video_embeds=MultiModalFieldConfig.flat_from_sizes("video", video_embed_sizes),
            video_grid_thw=MultiModalFieldConfig.batched("video"),
        )

    Qwen3VLMultiModalProcessor._get_data_parser = _patched_get_data_parser
    Qwen3VLMultiModalProcessor._get_mm_fields_config = _patched_get_mm_fields_config


def attach_local_vision_branch(model, *, initialize_from_global: bool = True):
    """Attach the crop encoder, optionally cloning global-vision weights.

    Inference passes ``initialize_from_global=False`` because a strict full
    local checkpoint is loaded immediately afterwards. Avoiding the temporary
    clone saves both the copy and an unnecessary FP32 local branch.
    """

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
        reference_parameter = next(core_model.visual.parameters())
        local_visual = type(core_model.visual)._from_config(
            core_model.config.vision_config,
            dtype=reference_parameter.dtype,
        )
        if reference_parameter.device.type != "meta":
            local_visual.to(
                device=reference_parameter.device,
                dtype=reference_parameter.dtype,
            )
        if initialize_from_global:
            local_visual.load_state_dict(core_model.visual.state_dict(), strict=True)
        core_model.local_visual = local_visual

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


def wrap_numbers_string(num_str: str) -> Tuple[str, str]:
    path_str = "_" + "_".join(num_str.strip().split())
    boxed = f"<|box_start|> {num_str.strip()} <|box_end|>"
    return path_str, boxed

def split_text_and_numbers(s: str) -> Tuple[str, Optional[str]]:
    s = s.rstrip()
    m = re.search(
        r"(?P<prefix><\|box_start\|>\s*)(?P<nums>-?\d+\s+-?\d+\s+-?\d+\s+-?\d+)\s*$",
        s,
    )
    if not m:
        return s, None
    prefix = m.group("prefix")
    num_str = m.group("nums")
    text_part = s[: m.start()] + prefix
    return text_part, num_str


def is_local_patch_path(path: Any) -> bool:
    if not isinstance(path, str) or not path:
        return False
    return bool(re.search(r"_(\d+_){3}\d+$", path))


def select_anchor_image_path(message: list) -> Optional[str]:
    if not message or not isinstance(message, list):
        return None
    content_list = message[-1].get("content", [])
    image_items = [
        item
        for item in content_list
        if isinstance(item, dict)
        and item.get("type") == "image"
        and isinstance(item.get("image"), str)
        and item.get("image")
    ]
    if not image_items:
        return None
    return image_items[1]["image"] if len(image_items) >= 2 else image_items[0]["image"]


def infer_image_branch_flags(message: list) -> List[int]:
    flags = []
    for turn in message:
        for item in turn.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image":
                path = item.get("image", "")
                flags.append(1 if is_local_patch_path(path) else 0)
    return flags


def _normalize_branch_flags(flags: List[int], num_images: int) -> List[int]:
    if num_images <= 0:
        if flags:
            raise ValueError(
                f"image_branch_flags were provided without images: flags={len(flags)}."
            )
        return []
    normalized = [int(x) for x in flags]
    if len(normalized) != num_images:
        raise ValueError(
            "image_branch_flags/image_grid_thw count mismatch: "
            f"flags={len(normalized)}, images={num_images}."
        )
    if any(flag not in (0, 1) for flag in normalized):
        raise ValueError("image_branch_flags must contain only 0 (global) or 1 (local).")
    return normalized


@torch.no_grad()
def encode_images_with_external_branches(
    vision_model: Qwen3VLForConditionalGeneration,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.LongTensor,
    image_branch_flags: List[int],
) -> Dict[str, Any]:
    core_model = vision_model.model
    global_visual = core_model.visual
    local_visual = getattr(core_model, "local_visual", global_visual)

    device = next(global_visual.parameters()).device
    local_param_device = next(local_visual.parameters()).device
    if local_param_device != device:
        local_visual.to(device=device, dtype=global_visual.dtype)

    pixel_values = pixel_values.to(device=device, dtype=global_visual.dtype)
    image_grid_thw = image_grid_thw.to(device=device)

    if image_grid_thw.ndim != 2 or image_grid_thw.shape[-1] != 3:
        raise ValueError(
            f"image_grid_thw must have shape [num_images, 3], got {tuple(image_grid_thw.shape)}."
        )
    expected_patch_rows = int(image_grid_thw.prod(-1).sum().item())
    if pixel_values.ndim < 1 or int(pixel_values.shape[0]) != expected_patch_rows:
        raise ValueError(
            "pixel_values/image_grid_thw patch-row mismatch: "
            f"pixel_rows={int(pixel_values.shape[0]) if pixel_values.ndim >= 1 else 0}, "
            f"grid_product_sum={expected_patch_rows}."
        )

    raw_split_sizes = image_grid_thw.prod(-1).tolist()
    pixel_splits = list(torch.split(pixel_values, raw_split_sizes))
    branch_flags = _normalize_branch_flags(image_branch_flags, image_grid_thw.shape[0])
    global_ids = [i for i, f in enumerate(branch_flags) if int(f) == 0]
    local_ids = [i for i, f in enumerate(branch_flags) if int(f) == 1]

    def _encode_group(sample_ids: List[int], visual_module):
        if len(sample_ids) == 0:
            return {}
        grouped_pixels = torch.cat([pixel_splits[i] for i in sample_ids], dim=0).to(dtype=visual_module.dtype)
        grouped_grid = torch.stack([image_grid_thw[i] for i in sample_ids], dim=0)
        base_cat, deepstack_cat = visual_module(grouped_pixels, grid_thw=grouped_grid)
        split_sizes = (grouped_grid.prod(-1) // visual_module.spatial_merge_size**2).tolist()
        base_split = list(torch.split(base_cat, split_sizes, dim=0))
        deepstack_split = [list(torch.split(layer_cat, split_sizes, dim=0)) for layer_cat in deepstack_cat]

        out = {}
        for k, sid in enumerate(sample_ids):
            parts = [base_split[k]]
            for layer_parts in deepstack_split:
                parts.append(layer_parts[k])
            out[sid] = torch.cat(parts, dim=-1)
        return out

    global_map = _encode_group(global_ids, global_visual)
    local_map = _encode_group(local_ids, local_visual)
    per_image_embeds = []
    for idx in range(len(branch_flags)):
        if idx in global_map:
            per_image_embeds.append(global_map[idx])
        else:
            per_image_embeds.append(local_map[idx])

    image_embeds = torch.cat(per_image_embeds, dim=0).cpu().contiguous()
    image_grid_thw = image_grid_thw.cpu().contiguous()
    return {"image": {"image_embeds": image_embeds, "image_grid_thw": image_grid_thw}}


def build_external_mm_data(
    processor: AutoProcessor,
    vision_model: Qwen3VLForConditionalGeneration,
    message: list,
    image_branch_flags: List[int],
) -> Dict[str, Any]:
    image_inputs: List[Any] = []
    for turn in message:
        for item in turn.get("content", []):
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            image = item.get("image")
            if image is None:
                continue
            if isinstance(image, (np.ndarray, Image.Image)):
                image_inputs.append(image)
                continue
            if isinstance(image, str):
                image_inputs.extend(process_one_new_image(processor, image))
                continue
            raise TypeError(f"Unsupported prompt image type for rollout encoding: {type(image)}")

    if not image_inputs:
        print("[UserRolloutBackend] build_external_mm_data no image inputs", flush=True)
        return {}

    image_features = processor.image_processor(images=image_inputs, return_tensors="pt")
    pixel_values = image_features.get("pixel_values", None)
    image_grid_thw = image_features.get("image_grid_thw", None)
    if pixel_values is None or image_grid_thw is None:
        print("[UserRolloutBackend] build_external_mm_data missing pixel_values or image_grid_thw", flush=True)
        return {}
    if image_grid_thw.ndim == 3 and image_grid_thw.shape[0] == 1:
        image_grid_thw = image_grid_thw[0]
    mm_data = encode_images_with_external_branches(
        vision_model=vision_model,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        image_branch_flags=image_branch_flags,
    )
    return mm_data


def process_one_new_image(processor: AutoProcessor, new_img_path: str):
    """Return the same unresized RGB crop used by SFT preprocessing."""
    # ``processor`` is intentionally applied once by the caller.
    image_path, bbox = parse_path_and_bbox(new_img_path)
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
    return [load_rgb_bbox(rgb_image, bbox)]


def append_external_mm_data_incremental(
    processor: AutoProcessor,
    vision_model: Qwen3VLForConditionalGeneration,
    prev_mm_data: Dict[str, Any],
    new_img_path: str,
) -> Optional[Dict[str, Any]]:
    new_image_inputs = process_one_new_image(processor, new_img_path)
    if len(new_image_inputs) == 0:
        return None
    new_features = processor.image_processor(images=new_image_inputs, return_tensors="pt")
    pixel_values = new_features.get("pixel_values", None)
    image_grid_thw = new_features.get("image_grid_thw", None)
    if pixel_values is None or image_grid_thw is None:
        return None
    if image_grid_thw.ndim == 3 and image_grid_thw.shape[0] == 1:
        image_grid_thw = image_grid_thw[0]

    new_mm_data = encode_images_with_external_branches(
        vision_model=vision_model,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        image_branch_flags=[1] * int(image_grid_thw.shape[0]),
    )

    return _merge_incremental_mm_data(prev_mm_data, new_mm_data)


def prepare_inputs_for_vllm_external(
    messages,
    processor: AutoProcessor,
    vision_model: Qwen3VLForConditionalGeneration,
    image_branch_flags: Optional[List[int]] = None,
):
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_token_ids = processor.tokenizer.encode(prompt, add_special_tokens=False)
    effective_image_branch_flags = (
        [int(flag) for flag in image_branch_flags]
        if image_branch_flags is not None
        else infer_image_branch_flags(messages)
    )
    mm_data = build_external_mm_data(processor, vision_model, messages, effective_image_branch_flags)
    return {
        "prompt": prompt,
        # Keep the unexpanded token prompt so continuation rounds can append
        # the exact vLLM output IDs instead of decode -> encode round-tripping.
        # vLLM's multimodal processor expands image placeholders on each call.
        "prompt_token_ids": list(prompt_token_ids),
        "multi_modal_data": mm_data,
        "image_branch_flags": effective_image_branch_flags,
    }


def _clone_rollout_messages(messages: list) -> list:
    cloned_messages: list = []
    for message in messages:
        if not isinstance(message, dict):
            cloned_messages.append(copy.deepcopy(message))
            continue
        cloned_message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            cloned_message["content"] = [
                dict(item) if isinstance(item, dict) else copy.deepcopy(item)
                for item in content
            ]
        elif isinstance(content, dict):
            cloned_message["content"] = dict(content)
        cloned_messages.append(cloned_message)
    return cloned_messages


def _merge_incremental_mm_data(
    prev_mm_data: Dict[str, Any],
    new_mm_data: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if new_mm_data is None:
        return None

    merged = dict(prev_mm_data) if prev_mm_data else {}
    old_image = merged.get("image")
    new_image = new_mm_data.get("image")
    if new_image is None:
        return merged
    if old_image is None:
        merged["image"] = new_image
        return merged

    merged["image"] = {
        "image_embeds": torch.cat([old_image["image_embeds"], new_image["image_embeds"]], dim=0).contiguous(),
        "image_grid_thw": torch.cat([old_image["image_grid_thw"], new_image["image_grid_thw"]], dim=0).contiguous(),
    }
    return merged


def _plan_incremental_append(text: str, anchor_path: str) -> Optional[Dict[str, str]]:
    _, num_str = split_text_and_numbers(text)
    if not num_str:
        print("[Warn] Incremental append skipped: no bbox numbers parsed from pause chunk.")
        return None
    if not anchor_path:
        print("[Warn] Incremental append skipped: anchor image path is missing.")
        return None

    path_str, _ = wrap_numbers_string(num_str)
    new_img_path = anchor_path + path_str
    boxed_text = f"{text}<|box_end|>"
    observation_text = "\n<|vision_start|><|image_pad|><|vision_end|>\n"
    return {
        "new_img_path": new_img_path,
        # Do not strip model output: even whitespace is part of the sampled
        # token trajectory and later action log-probabilities must use it.
        "boxed_text": boxed_text,
        "new_text": f"{boxed_text}{observation_text}",
        "add_text": f"{boxed_text}\n<image>\n",
        "observation_text": observation_text,
    }


def _split_batched_new_mm_data(
    vision_model: Qwen3VLForConditionalGeneration,
    batched_mm_data: Optional[Dict[str, Any]],
    image_counts: List[int],
) -> List[Optional[Dict[str, Any]]]:
    if batched_mm_data is None:
        return [None for _ in image_counts]

    image_payload = batched_mm_data.get("image")
    if image_payload is None:
        return [None for _ in image_counts]

    image_embeds = image_payload.get("image_embeds")
    image_grid_thw = image_payload.get("image_grid_thw")
    if image_embeds is None or image_grid_thw is None:
        return [None for _ in image_counts]

    if image_grid_thw.ndim == 3:
        image_grid_thw = image_grid_thw.reshape(-1, image_grid_thw.shape[-1])

    merge_area = int(vision_model.model.visual.spatial_merge_size) ** 2
    per_image_embed_sizes = (image_grid_thw.prod(-1) // merge_area).tolist()
    split_payloads: List[Optional[Dict[str, Any]]] = []
    grid_offset = 0
    embed_offset = 0
    for image_count in image_counts:
        if image_count <= 0:
            split_payloads.append(None)
            continue
        next_grid_offset = grid_offset + image_count
        sample_grid = image_grid_thw[grid_offset:next_grid_offset].contiguous()
        sample_embed_rows = int(sum(per_image_embed_sizes[grid_offset:next_grid_offset]))
        next_embed_offset = embed_offset + sample_embed_rows
        sample_embeds = image_embeds[embed_offset:next_embed_offset].contiguous()
        split_payloads.append(
            {
                "image": {
                    "image_embeds": sample_embeds,
                    "image_grid_thw": sample_grid,
                }
            }
        )
        grid_offset = next_grid_offset
        embed_offset = next_embed_offset

    if grid_offset != int(image_grid_thw.shape[0]) or embed_offset != int(image_embeds.shape[0]):
        raise RuntimeError(
            "Batched incremental mm_data split mismatch: "
            f"consumed_grids={grid_offset}/{int(image_grid_thw.shape[0])}, "
            f"consumed_embeds={embed_offset}/{int(image_embeds.shape[0])}."
        )

    return split_payloads


def _build_pause_update_payload_batch(
    idxs: List[int],
    round_payload: List[Dict[str, Any]],
    states: List["SampleState"],
    pause_token_id: int,
    processor: AutoProcessor,
    vision_model: Qwen3VLForConditionalGeneration,
    owner_rank_filter: Optional[int] = None,
    max_pause_times: Optional[int] = None,
    image_encode_batch_size: Optional[int] = None,
) -> List[Optional[Dict[str, Any]]]:
    if image_encode_batch_size is not None and int(image_encode_batch_size) <= 0:
        raise ValueError("image_encode_batch_size must be > 0 when provided.")

    pause_updates: List[Optional[Dict[str, Any]]] = [None for _ in idxs]
    batch_candidates: List[Tuple[int, Dict[str, str], Dict[str, Any]]] = []

    for payload_idx, (i, out_payload) in enumerate(zip(idxs, round_payload)):
        if out_payload.get("stopped_by") != pause_token_id:
            continue

        s = states[i]
        if owner_rank_filter is not None and s.owner_rank != owner_rank_filter:
            continue
        if max_pause_times is not None and s.pause_count >= int(max_pause_times):
            continue
        text = out_payload.get("text", "")
        plan = _plan_incremental_append(text, s.anchor_image)
        if plan is None:
            boxed_text = text+"<|box_end|>"
            pause_updates[payload_idx] = {
                "ok": False,
                "new_text": boxed_text,
                "add_text": None,
                "observation_text": None,
                "new_mm_data": s.mm_data,
                "new_img_path": None,
            }
            continue

        batch_candidates.append((payload_idx, plan, s.mm_data))

    if not batch_candidates:
        return pause_updates

    chunk_size = (
        len(batch_candidates)
        if image_encode_batch_size is None
        else int(image_encode_batch_size)
    )
    for chunk_start in range(0, len(batch_candidates), chunk_size):
        candidate_chunk = batch_candidates[chunk_start : chunk_start + chunk_size]
        valid_candidates: List[Tuple[int, Dict[str, str], Dict[str, Any]]] = []
        all_image_inputs: List[Any] = []
        image_counts: List[int] = []

        for payload_idx, plan, prev_mm_data in candidate_chunk:
            try:
                new_image_inputs = process_one_new_image(processor, plan["new_img_path"])
            except Exception as exc:
                if _is_out_of_memory_error(exc):
                    raise
                print(
                    "[UserRolloutBackend] skipping one invalid pause image: "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                new_image_inputs = []
            if len(new_image_inputs) == 0:
                pause_updates[payload_idx] = {
                    "ok": False,
                    "new_text": plan["boxed_text"],
                    "add_text": None,
                    "observation_text": None,
                    "new_mm_data": prev_mm_data,
                    "new_img_path": plan["new_img_path"],
                }
                continue

            valid_candidates.append((payload_idx, plan, prev_mm_data))
            image_counts.append(len(new_image_inputs))
            all_image_inputs.extend(new_image_inputs)

        if not valid_candidates:
            continue

        try:
            new_features = processor.image_processor(images=all_image_inputs, return_tensors="pt")
            pixel_values = new_features.get("pixel_values", None)
            image_grid_thw = new_features.get("image_grid_thw", None)
            if pixel_values is None or image_grid_thw is None:
                raise RuntimeError("processor.image_processor returned no pixel_values/image_grid_thw for pause batch.")
            if image_grid_thw.ndim == 3:
                image_grid_thw = image_grid_thw.reshape(-1, image_grid_thw.shape[-1])

            batched_new_mm_data = encode_images_with_external_branches(
                vision_model=vision_model,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                image_branch_flags=[1] * int(image_grid_thw.shape[0]),
            )
            split_new_mm_data = _split_batched_new_mm_data(
                vision_model=vision_model,
                batched_mm_data=batched_new_mm_data,
                image_counts=image_counts,
            )
        except Exception as exc:
            if _is_out_of_memory_error(exc):
                raise
            print(
                "[UserRolloutBackend] batched pause image encoding failed; falling back to per-sample processing: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            for payload_idx, plan, prev_mm_data in valid_candidates:
                try:
                    merged_mm_data = append_external_mm_data_incremental(
                        processor=processor,
                        vision_model=vision_model,
                        prev_mm_data=prev_mm_data,
                        new_img_path=plan["new_img_path"],
                    )
                except Exception as sample_exc:
                    if _is_out_of_memory_error(sample_exc):
                        raise
                    print(
                        "[UserRolloutBackend] skipping one pause after per-sample encoding failure: "
                        f"error={type(sample_exc).__name__}: {sample_exc}",
                        flush=True,
                    )
                    merged_mm_data = None
                pause_updates[payload_idx] = {
                    "ok": merged_mm_data is not None,
                    "new_text": plan["new_text"] if merged_mm_data is not None else plan["boxed_text"],
                    "add_text": plan["add_text"] if merged_mm_data is not None else None,
                    "observation_text": plan["observation_text"] if merged_mm_data is not None else None,
                    "new_mm_data": merged_mm_data,
                    "new_img_path": plan["new_img_path"],
                }
            continue

        for (payload_idx, plan, prev_mm_data), sample_new_mm_data in zip(
            valid_candidates, split_new_mm_data
        ):
            merged_mm_data = _merge_incremental_mm_data(prev_mm_data, sample_new_mm_data)
            pause_updates[payload_idx] = {
                "ok": merged_mm_data is not None,
                "new_text": plan["new_text"] if merged_mm_data is not None else plan["boxed_text"],
                "add_text": plan["add_text"] if merged_mm_data is not None else None,
                "observation_text": plan["observation_text"] if merged_mm_data is not None else None,
                "new_mm_data": merged_mm_data,
                "new_img_path": plan["new_img_path"],
            }

    return pause_updates


def _merge_partial_pause_update_payloads(
    gathered_payloads: List[Optional[List[Optional[Dict[str, Any]]]]],
    expected_len: int,
) -> List[Optional[Dict[str, Any]]]:
    merged: List[Optional[Dict[str, Any]]] = [None for _ in range(expected_len)]
    for payload in gathered_payloads:
        if payload is None:
            continue
        if len(payload) != expected_len:
            raise RuntimeError(
                "Gathered pause update payload length mismatch: "
                f"expected={expected_len}, got={len(payload)}."
            )
        for idx, item in enumerate(payload):
            if item is None:
                continue
            if merged[idx] is not None:
                raise RuntimeError(f"Duplicate pause update payload for group item index={idx}.")
            merged[idx] = item
    return merged


def build_append_data_incremental_external(
    message: list,
    text: str,
    anchor_path: str,
    image_branch_flags: List[int],
    processor: AutoProcessor,
    vision_model: Qwen3VLForConditionalGeneration,
    prev_mm_data: Dict[str, Any],
):
    plan = _plan_incremental_append(text, anchor_path)
    if plan is None:
        return text, None, prev_mm_data, False

    new_mm_data = append_external_mm_data_incremental(
        processor=processor,
        vision_model=vision_model,
        prev_mm_data=prev_mm_data,
        new_img_path=plan["new_img_path"],
    )
    if new_mm_data is None:
        return text, None, prev_mm_data, False

    content_list = message[-1].get("content", [])
    content_list.append({"type": "image", "image": plan["new_img_path"]})
    message[-1]["content"] = content_list
    image_branch_flags.append(1)
    return plan["new_text"], plan["add_text"], new_mm_data, True


def get_stop_reason_token_id(gen_out):
    stop_reason = getattr(gen_out, "stop_reason", None)
    if isinstance(stop_reason, int):
        return stop_reason
    if isinstance(stop_reason, str):
        match = re.search(r"(\d+)", stop_reason)
        if match:
            return int(match.group(1))
    return None


def _build_generation_payload(processor, gen_out, allowed_tokens: int) -> Dict[str, Any]:
    raw_token_ids = getattr(gen_out, "token_ids", None)
    if raw_token_ids is None:
        raise RuntimeError("vLLM completion did not return token_ids; exact GRPO action tracking is required.")
    try:
        token_ids = [int(token_id) for token_id in raw_token_ids]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("vLLM completion returned invalid token_ids.") from exc
    token_count = len(token_ids)
    finish_reason = getattr(gen_out, "finish_reason", None)
    stopped_by = get_stop_reason_token_id(gen_out)
    budget_clipped = False

    if token_count > allowed_tokens:
        clipped_token_ids = token_ids[:allowed_tokens]
        token_count = len(clipped_token_ids)
        finish_reason = "length"
        stopped_by = None
        budget_clipped = True
        token_ids = clipped_token_ids

    if stopped_by is not None and (not token_ids or token_ids[-1] != stopped_by):
        raise RuntimeError(
            "vLLM stop token is missing from completion token_ids: "
            f"stop_reason={stopped_by}, last_token={token_ids[-1] if token_ids else None}."
        )

    # Decode with the same explicitly validated processor used to build the
    # trainer trace.  vLLM's export tokenizer may carry legacy regex metadata;
    # prompt_token_ids/action token_ids remain authoritative.
    primary_eos = getattr(processor.tokenizer, "eos_token_id", None)
    terminal_token_included = bool(
        stopped_by is not None
        or (
            finish_reason == "stop"
            and token_ids
            and primary_eos is not None
            and token_ids[-1] == int(primary_eos)
        )
    )
    decoded_token_ids = token_ids[:-1] if terminal_token_included else token_ids
    text = processor.tokenizer.decode(
        decoded_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    return {
        "text": text,
        "token_ids": token_ids,
        "token_count": token_count,
        "finish_reason": finish_reason,
        "stopped_by": stopped_by,
        "budget_clipped": budget_clipped,
    }


def _external_mm_token_overhead(mm_data: Dict[str, Any]) -> int:
    """Return exact extra prompt tokens created by image-pad expansion.

    ``prompt_token_ids`` contains one logical image-pad token per image, while
    external ``image_embeds`` contains one row per expanded visual token.
    Their difference is the exact amount vLLM adds to the model sequence.
    """
    if not mm_data or not mm_data.get("image"):
        return 0
    image_payload = mm_data["image"]
    image_embeds = image_payload.get("image_embeds")
    image_grid_thw = image_payload.get("image_grid_thw")
    if not isinstance(image_embeds, torch.Tensor) or not isinstance(image_grid_thw, torch.Tensor):
        raise ValueError("External image mm_data must contain tensor image_embeds and image_grid_thw.")
    if image_embeds.ndim < 2:
        raise ValueError(f"image_embeds must be at least 2-D, got shape={tuple(image_embeds.shape)}.")
    if image_grid_thw.ndim == 1:
        if image_grid_thw.numel() % 3 != 0:
            raise ValueError(f"Invalid flattened image_grid_thw shape={tuple(image_grid_thw.shape)}.")
        image_count = int(image_grid_thw.numel() // 3)
    else:
        image_count = int(image_grid_thw.reshape(-1, image_grid_thw.shape[-1]).shape[0])
    visual_token_count = int(image_embeds.shape[0])
    if visual_token_count < image_count:
        raise ValueError(
            "External image embeddings contain fewer token rows than images: "
            f"visual_tokens={visual_token_count}, images={image_count}."
        )
    return visual_token_count - image_count


@dataclass
class SampleState:
    idx: int
    report_id: Any
    prompt: str
    prompt_token_ids: List[int]
    mm_data: Dict[str, Any]
    message: list
    image_branch_flags: List[int]
    anchor_image: Optional[str] = None
    all_text: str = ""
    all_token_ids: List[int] = field(default_factory=list)
    all_action_mask: List[int] = field(default_factory=list)
    pause_count: int = 0
    total_budget: int = 4096
    done: bool = False
    owner_rank: int = 0
    prompt_token_len: int = 0


def iterative_generate_dynamic_batch_external(
    llm,
    processor,
    vision_model,
    inputs: List[Dict[str, Any]],
    messages: List[list],
    report_list: List[dict],
    base_params,
    pause_token_id: int = 151649,
    true_stop_token_id: Optional[Any] = None,
    max_pause_times: int = 17,
    max_total_new_tokens: int = 4096,
    shared_generate_owner_rank: Optional[int] = None,
    shared_generate_rank: int = 0,
    shared_generate_group: Optional[Any] = None,
    shared_vision_owner_only: bool = False,
    state_owner_ranks: Optional[List[int]] = None,
    max_model_len: int = 4096,
    pause_image_encode_batch_size: Optional[int] = None,
):
    if (
        pause_image_encode_batch_size is not None
        and int(pause_image_encode_batch_size) <= 0
    ):
        raise ValueError("pause_image_encode_batch_size must be > 0 when provided.")

    if true_stop_token_id is None:
        true_stop_token_id = processor.tokenizer.eos_token_id
        if true_stop_token_id is None:
            raise ValueError("tokenizer.eos_token_id is None")

    raw_stop_ids = (
        list(true_stop_token_id)
        if isinstance(true_stop_token_id, (list, tuple, set))
        else [true_stop_token_id]
    )
    true_stop_token_ids = {int(token_id) for token_id in raw_stop_ids}
    if not true_stop_token_ids:
        raise ValueError("At least one true stop token ID is required.")

    states: List[SampleState] = []
    if state_owner_ranks is not None and len(state_owner_ranks) != len(inputs):
        raise ValueError(
            "state_owner_ranks length must match inputs length: "
            f"{len(state_owner_ranks)} != {len(inputs)}."
        )
    for i, (one_input, msg, rep) in enumerate(zip(inputs, messages, report_list)):
        prompt = one_input["prompt"]
        prompt_token_ids = [int(token_id) for token_id in one_input["prompt_token_ids"]]
        base_prompt_token_len = len(prompt_token_ids)
        mm_data = one_input.get("multi_modal_data", {}) or {}
        initial_visual_overhead = _external_mm_token_overhead(mm_data)
        states.append(
            SampleState(
                idx=i,
                report_id=rep.get("question_id", i),
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                mm_data=mm_data,
                message=msg,
                image_branch_flags=list(one_input.get("image_branch_flags", [])),
                anchor_image=one_input.get("anchor_image") or select_anchor_image_path(msg),
                all_text="",
                pause_count=0,
                total_budget=max_total_new_tokens,
                done=False,
                owner_rank=state_owner_ranks[i] if state_owner_ranks is not None else 0,
                prompt_token_len=base_prompt_token_len + initial_visual_overhead,
            )
        )

    active = {s.idx for s in states}
    round_idx = 0
    merge_budget_groups = _bool_env("QWENVL_GRPO_VLLM_MERGE_BUDGET_GROUPS", True)
    while active:
        round_idx += 1
        round_ids = []
        for i in list(active):
            s = states[i]
            if s.done or s.total_budget <= 0:
                s.done = True
                active.discard(i)
                continue
            round_ids.append(i)
        if not round_ids:
            break

        base_max = getattr(base_params, "max_tokens", 2048)
        per_sample_limits: Dict[int, int] = {}
        for i in round_ids:
            s = states[i]
            try:
                prompt_len = s.prompt_token_len
            except Exception as exc:
                print(
                    "[UserRolloutBackend] prompt length estimation failed; "
                    f"using budget-only limit for sample={s.report_id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                per_sample_limits[i] = min(base_max, s.total_budget)
                continue

            available_decode_budget = max_model_len - prompt_len
            if available_decode_budget <= 0:
                print(
                    "[UserRolloutBackend] stopping sample because prompt already reached "
                    f"max_model_len: sample={s.report_id}, prompt_len={prompt_len}, "
                    f"max_model_len={max_model_len}, pause_count={s.pause_count}",
                    flush=True,
                )
                s.done = True
                active.discard(i)
                continue

            per_sample_limits[i] = min(base_max, s.total_budget, available_decode_budget)

        if not per_sample_limits:
            continue

        groups: Dict[int, List[int]] = {}
        if merge_budget_groups:
            merged_max_tokens = max(per_sample_limits.values()) if per_sample_limits else base_max
            groups[merged_max_tokens] = list(per_sample_limits.keys())
        else:
            for i, per_round_max in per_sample_limits.items():
                groups.setdefault(per_round_max, []).append(i)

        for per_round_max, idxs in groups.items():
            params = SamplingParams(
                temperature=base_params.temperature,
                top_p=base_params.top_p,
                top_k=base_params.top_k,
                max_tokens=per_round_max,
                repetition_penalty=base_params.repetition_penalty,
                presence_penalty=base_params.presence_penalty,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
                seed=base_params.seed,
                stop_token_ids=[pause_token_id, *sorted(true_stop_token_ids)],
            )

            batch_inputs = []
            for i in idxs:
                if i not in per_sample_limits:
                    continue
                s = states[i]
                batch_inputs.append(
                    {
                        # Passing IDs makes each continuation round condition on
                        # the exact sampled prefix.  Supplying prompt text as
                        # metadata is safe because vLLM gives token IDs priority.
                        "prompt_token_ids": list(s.prompt_token_ids),
                        "prompt": s.prompt,
                        "multi_modal_data": s.mm_data,
                    }
                )
            idxs = [i for i in idxs if i in per_sample_limits]
            if not idxs:
                continue

            shared_generate = shared_generate_owner_rank is not None
            round_payload = None
            generation_error = None
            if (not shared_generate) or shared_generate_rank == shared_generate_owner_rank:
                try:
                    outs = llm.generate(batch_inputs, sampling_params=params, use_tqdm=False)
                    if len(outs) != len(idxs):
                        raise RuntimeError(
                            "vLLM returned a mismatched completion count: "
                            f"requests={len(idxs)}, outputs={len(outs)}."
                        )
                    round_payload = []
                    for sample_idx, out in zip(idxs, outs):
                        gen_out = out.outputs[0]
                        payload = _build_generation_payload(
                            processor,
                            gen_out,
                            allowed_tokens=per_sample_limits[sample_idx],
                        )
                        round_payload.append(payload)
                except Exception as exc:
                    generation_error = (
                        f"{type(exc).__name__}: {exc}\n"
                        + traceback.format_exc(limit=20)
                    )
            if shared_generate:
                payload_box = [
                    {
                        "payload": round_payload,
                        "error": generation_error,
                    }
                    if shared_generate_rank == shared_generate_owner_rank
                    else None
                ]
                dist.broadcast_object_list(
                    payload_box,
                    src=shared_generate_owner_rank,
                    group=shared_generate_group,
                )
                shared_payload = payload_box[0]
                if not isinstance(shared_payload, dict):
                    raise RuntimeError("Shared rollout generate broadcast returned an invalid envelope.")
                if shared_payload.get("error"):
                    raise RuntimeError(
                        "Shared rollout generation failed on the owner rank:\n"
                        + str(shared_payload["error"])
                    )
                round_payload = shared_payload.get("payload")
                if round_payload is None:
                    raise RuntimeError("Shared rollout generate broadcast returned no payload.")
            elif generation_error is not None:
                raise RuntimeError("Rollout generation failed:\n" + generation_error)
            if len(round_payload) != len(idxs):
                raise RuntimeError(
                    "Rollout payload/request count mismatch: "
                    f"requests={len(idxs)}, payloads={len(round_payload)}."
                )

            pause_update_payload = None
            has_pause_updates = any(
                out_payload.get("stopped_by") == pause_token_id
                and states[i].pause_count < max_pause_times
                for i, out_payload in zip(idxs, round_payload)
            )
            if has_pause_updates:
                if shared_generate and shared_vision_owner_only:
                    pause_update_error = None
                    if shared_generate_rank == shared_generate_owner_rank:
                        try:
                            pause_update_payload = _build_pause_update_payload_batch(
                                idxs=idxs,
                                round_payload=round_payload,
                                states=states,
                                pause_token_id=pause_token_id,
                                processor=processor,
                                vision_model=vision_model,
                                max_pause_times=max_pause_times,
                                image_encode_batch_size=pause_image_encode_batch_size,
                            )
                        except Exception as exc:
                            pause_update_error = (
                                f"{type(exc).__name__}: {exc}\n"
                                + traceback.format_exc(limit=20)
                            )
                    pause_box = [
                        {
                            "payload": pause_update_payload,
                            "error": pause_update_error,
                        }
                        if shared_generate_rank == shared_generate_owner_rank
                        else None
                    ]
                    dist.broadcast_object_list(
                        pause_box,
                        src=shared_generate_owner_rank,
                        group=shared_generate_group,
                    )
                    shared_pause_payload = pause_box[0]
                    if not isinstance(shared_pause_payload, dict):
                        raise RuntimeError("Shared rollout pause-update broadcast returned an invalid envelope.")
                    if shared_pause_payload.get("error"):
                        raise RuntimeError(
                            "Shared rollout pause-update failed on the owner rank:\n"
                            + str(shared_pause_payload["error"])
                        )
                    pause_update_payload = shared_pause_payload.get("payload")
                    if pause_update_payload is None:
                        raise RuntimeError("Shared rollout pause-update broadcast returned no payload.")
                else:
                    if shared_generate and state_owner_ranks is not None:
                        pause_update_payload = _build_pause_update_payload_batch(
                            idxs=idxs,
                            round_payload=round_payload,
                            states=states,
                            pause_token_id=pause_token_id,
                            processor=processor,
                            vision_model=vision_model,
                            owner_rank_filter=shared_generate_rank,
                            max_pause_times=max_pause_times,
                            image_encode_batch_size=pause_image_encode_batch_size,
                        )
                        gathered_pause_payloads: List[Optional[List[Optional[Dict[str, Any]]]]] = [
                            None for _ in range(_get_world_size())
                        ]
                        dist.all_gather_object(
                            gathered_pause_payloads,
                            pause_update_payload,
                            group=shared_generate_group,
                        )
                        pause_update_payload = _merge_partial_pause_update_payloads(
                            gathered_pause_payloads,
                            expected_len=len(idxs),
                        )
                    else:
                        pause_update_payload = _build_pause_update_payload_batch(
                            idxs=idxs,
                            round_payload=round_payload,
                            states=states,
                            pause_token_id=pause_token_id,
                            processor=processor,
                            vision_model=vision_model,
                            max_pause_times=max_pause_times,
                            image_encode_batch_size=pause_image_encode_batch_size,
                        )

            for payload_idx, (i, out_payload) in enumerate(zip(idxs, round_payload)):
                s = states[i]
                text = out_payload.get("text", "")
                action_token_ids = out_payload.get("token_ids")
                if not isinstance(action_token_ids, list):
                    raise RuntimeError("Rollout payload is missing exact action token_ids.")
                action_token_ids = [int(token_id) for token_id in action_token_ids]
                s.all_token_ids.extend(action_token_ids)
                s.all_action_mask.extend([1] * len(action_token_ids))

                token_count = out_payload.get("token_count")
                if isinstance(token_count, int):
                    s.total_budget -= token_count

                finish_reason = out_payload.get("finish_reason")
                stopped_by = out_payload.get("stopped_by")

                if stopped_by == pause_token_id:
                    # Allow the model to produce a final answer after the last
                    # permitted observation, but do not execute one extra crop.
                    if s.pause_count >= max_pause_times:
                        s.all_text += processor.tokenizer.decode(
                            action_token_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        s.done = True
                        active.discard(i)
                        continue
                    pause_update = (
                        pause_update_payload[payload_idx]
                        if pause_update_payload is not None and payload_idx < len(pause_update_payload)
                        else None
                    )
                    if pause_update is None:
                        raise RuntimeError("Missing rollout pause update payload.")
                    ok = bool(pause_update.get("ok"))
                    new_text = pause_update.get("new_text")
                    add_text = pause_update.get("add_text")
                    observation_text = pause_update.get("observation_text")
                    new_mm_data = pause_update.get("new_mm_data")
                    new_img_path = pause_update.get("new_img_path")
                    if not ok:
                        if isinstance(new_text, str) and new_text:
                            s.all_text += new_text
                        s.done = True
                        active.discard(i)
                        continue

                    if new_img_path:
                        content_list = s.message[-1].get("content", [])
                        content_list.append({"type": "image", "image": new_img_path})
                        s.message[-1]["content"] = content_list
                        s.image_branch_flags.append(1)
                    if not isinstance(observation_text, str) or not observation_text:
                        raise RuntimeError("Successful rollout pause update is missing observation_text.")
                    observation_token_ids = processor.tokenizer.encode(
                        observation_text,
                        add_special_tokens=False,
                    )
                    s.prompt = s.prompt + new_text
                    s.prompt_token_ids.extend(action_token_ids)
                    s.prompt_token_ids.extend(observation_token_ids)
                    s.all_token_ids.extend(observation_token_ids)
                    s.all_action_mask.extend([0] * len(observation_token_ids))
                    s.all_text += add_text
                    s.mm_data = new_mm_data
                    s.prompt_token_len = len(s.prompt_token_ids) + _external_mm_token_overhead(s.mm_data)
                    s.pause_count += 1
                    continue

                s.all_text += text

                if stopped_by in true_stop_token_ids:
                    s.done = True
                    active.discard(i)
                    continue

                if finish_reason in ("length", "abort"):
                    s.done = True
                    active.discard(i)
                    continue
                # vLLM may add EOS IDs from generation_config (for Qwen3-VL,
                # 151643 in addition to tokenizer.eos_token_id=151645).  Any
                # non-pause stop is terminal even if it is not the primary EOS.
                if finish_reason == "stop":
                    s.done = True
                    active.discard(i)
                    continue
                if not text:
                    s.done = True
                    active.discard(i)
                    continue

                # Defensive continuation for a backend-specific nonterminal
                # finish reason: condition the next round on the exact action
                # prefix instead of sampling again from the old prompt.
                s.prompt = s.prompt + text
                s.prompt_token_ids.extend(action_token_ids)
                s.prompt_token_len = len(s.prompt_token_ids) + _external_mm_token_overhead(s.mm_data)
    return [
        {
            "id": s.report_id,
            "text": s.all_text,
            "token_ids": list(s.all_token_ids),
            "action_mask": list(s.all_action_mask),
        }
        for s in states
    ]


class UserRolloutBackend(BaseRolloutBackend):
    """vLLM rollout backend with external vision encoding and iterative crop expansion."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_name_or_path = kwargs.get("model_name_or_path")
        self.policy_model = kwargs.get("policy_model")
        self.policy_processor = kwargs.get("processor")
        self.training_args = kwargs.get("training_args")
        self.grpo_args = kwargs.get("grpo_args")

        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.rank = _get_rank()
        self.world_size = _get_world_size()

        if not self.model_name_or_path:
            raise ValueError("UserRolloutBackend requires `model_name_or_path` for initial vLLM bootstrap.")
        if self.policy_model is None:
            raise ValueError("UserRolloutBackend requires policy_model to reuse policy visual branches.")

        self.vllm_model_path = os.environ.get("QWENVL_GRPO_VLLM_MODEL_PATH", self.model_name_or_path)
                
        self.tensor_parallel_size = int(os.environ.get("QWENVL_GRPO_VLLM_TP_SIZE", "1"))
        self.engine_owner_rank = int(os.environ.get("QWENVL_GRPO_VLLM_OWNER_RANK", "0"))
        self.shared_vllm = bool(_dist_is_initialized() and self.tensor_parallel_size > 1)
        # Keep rollout control/object collectives off the trainer NCCL process
        # group. Non-owner ranks wait for owner-side vLLM generation on this
        # CPU/Gloo group, leaving the GPUs available to the vLLM TP workers.
        # Training, ZeRO gathers, and policy-to-vLLM weight sync intentionally
        # continue to use the default NCCL process group.
        self.rollout_control_group = (
            dist.new_group(ranks=list(range(self.world_size)), backend="gloo")
            if self.shared_vllm
            else None
        )
        self.shared_vision_owner_only = bool(
            self.shared_vllm and _bool_env("QWENVL_GRPO_SHARED_VISION_OWNER_ONLY", True)
        )
        self.shared_prepare_all_ranks = bool(
            self.shared_vllm and _bool_env("QWENVL_GRPO_SHARED_PREPARE_ALL_RANKS", False)
        )
        self.gpu_memory_utilization = float(os.environ.get("QWENVL_GRPO_VLLM_GPU_MEM_UTIL", "0.80"))
        env_num_gpu_blocks_override = os.environ.get(
            "QWENVL_GRPO_VLLM_NUM_GPU_BLOCKS_OVERRIDE"
        )
        self.num_gpu_blocks_override = (
            int(env_num_gpu_blocks_override)
            if env_num_gpu_blocks_override is not None
            else None
        )
        self.mm_processor_cache_gb = float(
            os.environ.get("QWENVL_GRPO_VLLM_MM_PROCESSOR_CACHE_GB", "0")
        )
        self.limit_mm_image = int(os.environ.get("QWENVL_GRPO_VLLM_LIMIT_MM_IMAGE", "256"))
        self.top_k = int(os.environ.get("QWENVL_GRPO_VLLM_TOP_K", "-1"))
        self.repetition_penalty = float(os.environ.get("QWENVL_GRPO_VLLM_REPETITION_PENALTY", "1.0"))
        self.presence_penalty = float(os.environ.get("QWENVL_GRPO_VLLM_PRESENCE_PENALTY", "0.0"))
        self.max_pause_times = int(os.environ.get("QWENVL_GRPO_MAX_PAUSE_TIMES", "17"))
        self.pause_token_id = int(os.environ.get("QWENVL_GRPO_PAUSE_TOKEN_ID", "151649"))
        # Use a fixed default for reproducible rollouts.  An explicit
        # QWENVL_GRPO_VLLM_SEED still takes precedence.
        self.seed = int(os.environ.get("QWENVL_GRPO_VLLM_SEED", "3407"))
        env_max_model_len = os.environ.get("QWENVL_GRPO_VLLM_MAX_MODEL_LEN")
        self.vllm_max_model_len = int(
            env_max_model_len
            or getattr(self.grpo_args, "vllm_max_model_len", 0)
            or 4096
        )

        self.sync_interval_steps = int(os.environ.get("QWENVL_GRPO_SYNC_INTERVAL_STEPS", "1"))
        self.weight_transfer_backend = (
            os.environ.get("QWENVL_GRPO_VLLM_WEIGHT_TRANSFER_BACKEND", "ipc")
            .strip()
            .lower()
        )
        self.ipc_chunk_mb = max(1.0, float(os.environ.get("QWENVL_GRPO_VLLM_IPC_CHUNK_MB", "64")))
        self.sync_verify = _bool_env("QWENVL_GRPO_VLLM_SYNC_VERIFY", False)
        # -1 means the bootstrap weights have not yet been verified/replaced
        # from the actual policy.  Step 0 must sync before the first rollout.
        self._last_synced_step = -1

        self._weight_transfer_ready = False

        self.processor = None
        self.vision_model = None
        self.llm = None
        self._runtime_ready = False
        self._llm_kwargs = None

        if self.tensor_parallel_size <= 0:
            raise ValueError("QWENVL_GRPO_VLLM_TP_SIZE must be > 0.")
        if self.world_size > 1 and self.tensor_parallel_size == 1:
            raise ValueError(
                "Multi-rank training with TP=1 would start one unsynchronized vLLM engine per rank; "
                "use shared vLLM with QWENVL_GRPO_VLLM_TP_SIZE>1."
            )
        if not (0.0 < self.gpu_memory_utilization < 1.0):
            raise ValueError("QWENVL_GRPO_VLLM_GPU_MEM_UTIL must be in (0, 1).")
        if self.num_gpu_blocks_override is not None and self.num_gpu_blocks_override <= 0:
            raise ValueError("QWENVL_GRPO_VLLM_NUM_GPU_BLOCKS_OVERRIDE must be > 0.")
        if self.mm_processor_cache_gb < 0.0:
            raise ValueError("QWENVL_GRPO_VLLM_MM_PROCESSOR_CACHE_GB must be >= 0.")
        if self.limit_mm_image <= 0:
            raise ValueError("QWENVL_GRPO_VLLM_LIMIT_MM_IMAGE must be > 0.")
        if self.max_pause_times < 0:
            raise ValueError("QWENVL_GRPO_MAX_PAUSE_TIMES must be >= 0.")
        if self.vllm_max_model_len <= 0:
            raise ValueError("vLLM max_model_len must be > 0.")
        if self.sync_interval_steps != 1:
            raise ValueError(
                "QWENVL_GRPO_SYNC_INTERVAL_STEPS must be 1; sampling from a stale rollout "
                "policy while computing old_log_probs with the current policy is invalid for GRPO."
            )
        if self.shared_vllm and self.engine_owner_rank < 0:
            raise ValueError("QWENVL_GRPO_VLLM_OWNER_RANK must be >= 0.")
        if self.shared_vllm and self.engine_owner_rank >= self.world_size:
            raise ValueError(
                f"QWENVL_GRPO_VLLM_OWNER_RANK={self.engine_owner_rank} is out of range for world_size={self.world_size}."
            )
        if self.shared_vllm and self.tensor_parallel_size > torch.cuda.device_count():
            raise ValueError(
                f"QWENVL_GRPO_VLLM_TP_SIZE={self.tensor_parallel_size} exceeds visible CUDA devices="
                f"{torch.cuda.device_count()}."
            )
        if self.weight_transfer_backend != "ipc":
            raise ValueError(
                "QWENVL_GRPO_VLLM_WEIGHT_TRANSFER_BACKEND must be 'ipc'; "
                f"got {self.weight_transfer_backend!r}."
            )
        if self.shared_vllm and self.rank == self.engine_owner_rank:
            print(
                f"[UserRolloutBackend] shared vLLM enabled: owner_rank={self.engine_owner_rank}, "
                f"tensor_parallel_size={self.tensor_parallel_size}, world_size={self.world_size}",
                flush=True,
            )
            print(
                "[UserRolloutBackend] dedicated Gloo rollout control group enabled; "
                "training/ZeRO/weight-sync collectives remain on NCCL.",
                flush=True,
            )
            if self.shared_vision_owner_only:
                print(
                    "[UserRolloutBackend] shared external vision owner-only mode enabled; "
                    "owner rank will build mm_data and broadcast to peer ranks.",
                    flush=True,
                )
            if self.shared_prepare_all_ranks:
                print(
                    "[UserRolloutBackend] shared rollout parallel-prepare mode enabled; "
                    "all ranks will build local mm_data and merge into one shared generate batch.",
                    flush=True,
                )

    def _build_detached_rollout_vision_model(
        self,
        materialize: bool = True,
    ) -> Optional[torch.nn.Module]:
        policy_model = self._unwrap_policy_model(self.policy_model)
        if not hasattr(policy_model, "model") or not hasattr(policy_model.model, "visual"):
            raise ValueError("Detached rollout vision model requires policy_model.model.visual to exist.")

        policy_core = policy_model.model
        visual_cls = type(policy_core.visual)
        visual_dtype = next(policy_core.visual.parameters()).dtype
        source_local_visual = getattr(policy_core, "local_visual", None)
        trainable_visual = [
            name
            for name, param in policy_core.named_parameters()
            if (name.startswith("visual.") or name.startswith("local_visual."))
            and param.requires_grad
        ]
        if trainable_visual:
            raise ValueError(
                "UserRolloutBackend keeps a detached visual snapshot and therefore requires "
                "global/local visual branches to be frozen during GRPO. Trainable examples: "
                + ", ".join(trainable_visual[:5])
            )

        with _temporarily_disable_deepspeed_zero_init():
            # Every ZeRO rank must enter GatheredParameters in the same order.
            # In owner-only mode peers participate in the gather but avoid
            # materializing a duplicate visual tower.
            with self._shared_visual_gather_context(vision_model=policy_model):
                if materialize:
                    visual = visual_cls(policy_core.config.vision_config)
                    visual.load_state_dict(policy_core.visual.state_dict(), strict=True)

                    local_visual = None
                    if source_local_visual is not None:
                        local_visual = visual_cls(policy_core.config.vision_config)
                        local_visual.load_state_dict(source_local_visual.state_dict(), strict=True)

        if not materialize:
            return None

        for module in [visual, local_visual]:
            if module is None:
                continue
            module.to(device=self.device, dtype=visual_dtype)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        rollout_vision_model = _RolloutVisionModel(
            _RolloutVisionCore(
                visual=visual,
                local_visual=local_visual,
            )
        )
        rollout_vision_model.eval()
        if self.rank == self.engine_owner_rank:
            print(
                "[UserRolloutBackend] loaded detached rollout vision model from current policy visual parameters "
                f"(has_local_visual={int(source_local_visual is not None)})",
                flush=True,
            )
        return rollout_vision_model

    def _validate_rollout_processor_matches_policy(self) -> None:
        if self.policy_processor is None:
            return
        policy_tokenizer = getattr(self.policy_processor, "tokenizer", self.policy_processor)
        rollout_tokenizer = getattr(self.processor, "tokenizer", self.processor)

        mismatches: List[str] = []
        for attr in ("eos_token_id", "pad_token_id", "bos_token_id"):
            policy_value = getattr(policy_tokenizer, attr, None)
            rollout_value = getattr(rollout_tokenizer, attr, None)
            if policy_value != rollout_value:
                mismatches.append(f"tokenizer.{attr}: policy={policy_value}, rollout={rollout_value}")

        special_tokens = (
            "<|im_start|>",
            "<|im_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|box_start|>",
            "<|box_end|>",
        )
        for token in special_tokens:
            policy_id = policy_tokenizer.convert_tokens_to_ids(token)
            rollout_id = rollout_tokenizer.convert_tokens_to_ids(token)
            if policy_id != rollout_id:
                mismatches.append(f"special token {token}: policy={policy_id}, rollout={rollout_id}")

        probe = (
            "<|im_start|>assistant\n<think>probe</think>\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n<|im_end|>"
        )
        policy_probe = policy_tokenizer.encode(probe, add_special_tokens=False)
        rollout_probe = rollout_tokenizer.encode(probe, add_special_tokens=False)
        if policy_probe != rollout_probe:
            mismatches.append("representative special/text tokenization differs")

        template_probe_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "unused-for-rendering.jpg"},
                    {"type": "text", "text": "probe"},
                ],
            }
        ]
        policy_rendered = self.policy_processor.apply_chat_template(
            template_probe_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        rollout_rendered = self.processor.apply_chat_template(
            template_probe_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if policy_rendered != rollout_rendered:
            mismatches.append("rendered multimodal chat template differs")
        elif policy_tokenizer.encode(policy_rendered, add_special_tokens=False) != rollout_tokenizer.encode(
            rollout_rendered,
            add_special_tokens=False,
        ):
            mismatches.append("rendered chat-template token IDs differ")

        policy_image_processor = getattr(self.policy_processor, "image_processor", None)
        rollout_image_processor = getattr(self.processor, "image_processor", None)
        if (policy_image_processor is None) != (rollout_image_processor is None):
            mismatches.append("image_processor presence differs")
        elif policy_image_processor is not None:
            for attr in ("min_pixels", "max_pixels", "patch_size", "merge_size", "temporal_patch_size"):
                policy_value = getattr(policy_image_processor, attr, None)
                rollout_value = getattr(rollout_image_processor, attr, None)
                if policy_value != rollout_value:
                    mismatches.append(
                        f"image_processor.{attr}: policy={policy_value}, rollout={rollout_value}"
                    )

        if mismatches:
            raise ValueError(
                "Rollout export processor is incompatible with the trainer processor; "
                "token traces/visual grids would not align. "
                + " | ".join(mismatches)
            )

    def _ensure_runtime(self):
        if self._runtime_ready:
            return

        print(
            f"[UserRolloutBackend] initializing runtime on rank={self.rank} "
            f"with vllm_model_path={self.vllm_model_path}",
            flush=True,
        )
        check_vllm_mm_embed_capability()
        patch_vllm_qwen3vl_mm_embed_field_sizes()

        print("[UserRolloutBackend] loading rollout processor", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            self.vllm_model_path,
            fix_mistral_regex=True,
        )
        self._validate_rollout_processor_matches_policy()
        if self.shared_vllm:
            if self.shared_vision_owner_only and (not self.shared_prepare_all_ranks):
                self.vision_model = self._build_detached_rollout_vision_model(
                    materialize=self.rank == self.engine_owner_rank,
                )
            else:
                self.vision_model = self._build_detached_rollout_vision_model()
        else:
            self.vision_model = self._unwrap_policy_model(self.policy_model)
            self.vision_model.eval()

        self._llm_kwargs = dict(
            model=self.vllm_model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            # External image embeddings are unique rollout observations, so
            # processor-cache reuse provides no benefit and retains large CPU
            # tensors across requests.
            mm_processor_cache_gb=self.mm_processor_cache_gb,
            limit_mm_per_prompt={"image": self.limit_mm_image},
            enable_expert_parallel=False,
            seed=self.seed,
            max_model_len=self.vllm_max_model_len,
            enable_mm_embeds=True,
            # Reuse identical/growing prefixes within one rollout step.  The
            # cache is explicitly reset after every completed policy-weight
            # sync so KV tensors never cross policy versions.
            enable_prefix_caching=True,
        )
        if self.num_gpu_blocks_override is not None:
            self._llm_kwargs["num_gpu_blocks_override"] = self.num_gpu_blocks_override
        # enforce_eager=True,

        try:
            from vllm.config import WeightTransferConfig

            self._llm_kwargs["weight_transfer_config"] = WeightTransferConfig(backend=self.weight_transfer_backend)
        except Exception as exc:
            raise RuntimeError(
                "Weight sync requested but WeightTransferConfig is unavailable for backend="
                f"{self.weight_transfer_backend!r}: {exc}"
            )

        if self.shared_vllm and self.rank != self.engine_owner_rank:
            print(
                f"[UserRolloutBackend] rank={self.rank} prepared rollout processor"
                f"{'' if self.shared_vision_owner_only else '/vision model'} and will reuse "
                f"shared vLLM from owner_rank={self.engine_owner_rank}",
                flush=True,
            )
            self._runtime_ready = True
            return

        print(
            f"[UserRolloutBackend] creating vLLM engine "
            f"(tp={self.tensor_parallel_size}, gpu_mem_util={self.gpu_memory_utilization}, "
            f"max_model_len={self.vllm_max_model_len}, "
            f"num_gpu_blocks_override={self.num_gpu_blocks_override})",
            flush=True,
        )
        if self.sync_verify:
            print(
                "[UserRolloutBackend] worker-side sync debug counters are disabled for IPC startup; "
                "update_weights success/failure remains enforced by the trainer.",
                flush=True,
            )
        with _temporarily_scrub_torchrun_env() as removed_env_keys:
            env_override_ctx = (
                _temporarily_set_env("QWENVL_GRPO_VLLM_SYNC_VERIFY", "0")
                if self.sync_verify
                else contextlib.nullcontext()
            )
            with env_override_ctx:
                if removed_env_keys:
                    print(
                        "[UserRolloutBackend] scrubbing outer torchrun env for vLLM spawn: "
                        + ", ".join(sorted(removed_env_keys)),
                        flush=True,
                    )
                self.llm = LLM(**self._llm_kwargs)
        print("[UserRolloutBackend] vLLM runtime ready", flush=True)
        self._runtime_ready = True

    def _build_sampling_params(self, request: RolloutRequest) -> SamplingParams:
        return SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=self.top_k,
            max_tokens=request.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            # Each expanded round starts from the same explicit request seed
            # for reproducible rollouts.
            seed=self.seed,
        )

    def _get_true_stop_token_ids(self) -> List[int]:
        stop_ids: List[int] = []
        tokenizer_eos = getattr(self.processor.tokenizer, "eos_token_id", None)
        if tokenizer_eos is not None:
            stop_ids.append(int(tokenizer_eos))

        policy_model = self._unwrap_policy_model(self.policy_model)
        generation_config = getattr(policy_model, "generation_config", None)
        configured_eos = getattr(generation_config, "eos_token_id", None)
        configured_values = (
            list(configured_eos)
            if isinstance(configured_eos, (list, tuple, set))
            else [configured_eos]
        )
        for token_id in configured_values:
            if token_id is not None:
                stop_ids.append(int(token_id))

        deduplicated = list(dict.fromkeys(stop_ids))
        if not deduplicated:
            raise ValueError("No EOS token IDs are available for rollout termination.")
        return deduplicated

    @staticmethod
    def _unwrap_policy_model(policy_model):
        return policy_model.module if hasattr(policy_model, "module") else policy_model

    def _get_rollout_vision_model(self):
        if self.vision_model is not None:
            return self.vision_model
        if (
            self.shared_vllm
            and self.shared_vision_owner_only
            and (not self.shared_prepare_all_ranks)
            and self.rank != self.engine_owner_rank
        ):
            return None
        return self._unwrap_policy_model(self.policy_model)

    def _iter_visual_parameters(self, vision_model=None):
        if vision_model is None:
            vision_model = self._get_rollout_vision_model()
        if vision_model is None or not hasattr(vision_model, "model"):
            return []
        core_model = vision_model.model
        params = []
        seen = set()
        for module in (getattr(core_model, "visual", None), getattr(core_model, "local_visual", None)):
            if module is None:
                continue
            for param in module.parameters():
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                params.append(param)
        return params

    def _shared_visual_gather_context(self, vision_model=None):
        stack = contextlib.ExitStack()

        params = self._iter_visual_parameters(vision_model=vision_model)
        if not params:
            return stack

        needs_gather = any(
            hasattr(param, "ds_id") or (hasattr(param, "ds_shape") and param.ds_shape is not None)
            for param in params
        )
        if not needs_gather:
            return stack

        try:
            import deepspeed
        except Exception as exc:
            raise RuntimeError(
                f"Shared rollout visual gather requires DeepSpeed for ZeRO-sharded params: {exc}"
            ) from exc

        if self.rank == self.engine_owner_rank:
            print(
                f"[UserRolloutBackend] gathering ZeRO-sharded visual parameters for shared rollout "
                f"(params={len(params)})",
                flush=True,
            )
        stack.enter_context(deepspeed.zero.GatheredParameters(params, modifier_rank=self.engine_owner_rank))
        return stack

    def _iter_sync_named_parameters(self, policy_model):
        src = self._unwrap_policy_model(policy_model)
        only_trainable = bool(int(os.environ.get("QWENVL_GRPO_SYNC_ONLY_TRAINABLE", "1")))
        for name, p in src.named_parameters():
            if name.startswith("model.visual.") or name.startswith("model.local_visual."):
                continue
            if only_trainable and (not p.requires_grad):
                continue
            yield name, p

    def _zero_gather_context(self, param):
        needs_gather = hasattr(param, "ds_id") or (hasattr(param, "ds_shape") and param.ds_shape is not None)
        if not needs_gather:
            return contextlib.nullcontext()
        try:
            import deepspeed
        except Exception as exc:
            raise RuntimeError(f"ZeRO-sharded parameter detected but deepspeed is unavailable: {exc}") from exc
        return deepspeed.zero.GatheredParameters([param], modifier_rank=self.engine_owner_rank)

    def _participate_in_zero_gather_for_sync(self, policy_model):
        first_error: Optional[str] = None
        for name, p in self._iter_sync_named_parameters(policy_model):
            try:
                with self._zero_gather_context(p):
                    pass
            except Exception as exc:
                if first_error is None:
                    first_error = (
                        f"parameter={name} {type(exc).__name__}: {exc}\n"
                        + traceback.format_exc(limit=20)
                    )
                # Match the owner: even after a local gather failure, enter
                # every remaining per-parameter gather in the same order.
                continue
        if first_error is not None:
            raise RuntimeError(
                "Peer ZeRO gather failed after completing the gather sequence:\n" + first_error
            )

    def _prepare_zero_for_sync(self, policy_model):
        optimizer = getattr(policy_model, "optimizer", None)
        parameter_offload = getattr(optimizer, "parameter_offload", None)
        if parameter_offload is None:
            return
        partition_all_parameters = getattr(parameter_offload, "partition_all_parameters", None)
        if partition_all_parameters is None:
            return
        partition_all_parameters()

    def _maybe_init_ipc_weight_transfer_engine(self):
        if self._weight_transfer_ready:
            return True
        if self.shared_vllm and self.rank != self.engine_owner_rank:
            return True

        try:
            print("[UserRolloutBackend] initializing vLLM weight transfer backend (ipc)", flush=True)
            self.llm.init_weight_transfer_engine(dict(init_info={}))
        except Exception as exc:
            raise RuntimeError(
                "IPC init_weight_transfer_engine failed: "
                f"{type(exc).__name__}: {exc!r}"
            ) from exc

        self._weight_transfer_ready = True
        return True

    def _get_ipc_target_device_indices(self) -> List[int]:
        return list(range(max(1, int(self.tensor_parallel_size))))

    @staticmethod
    def _get_cuda_device_uuid(device_index: int) -> str:
        return str(torch.cuda.get_device_properties(device_index).uuid)

    @staticmethod
    def _get_ipc_atomic_group_key(name: str) -> Optional[str]:
        qkv_match = re.match(r"^(.*\.(?:self_attn|attn))\.(q_proj|k_proj|v_proj)\.weight$", name)
        if qkv_match is not None:
            return f"{qkv_match.group(1)}.qkv_proj.weight"

        gate_up_match = re.match(r"^(.*\.mlp)\.(gate_proj|up_proj)\.weight$", name)
        if gate_up_match is not None:
            return f"{gate_up_match.group(1)}.gate_up_proj.weight"

        return None

    @staticmethod
    def _get_ipc_atomic_group_expected_size(name: str) -> int:
        if re.match(r"^.*\.(?:self_attn|attn)\.(q_proj|k_proj|v_proj)\.weight$", name):
            return 3
        if re.match(r"^.*\.mlp\.(gate_proj|up_proj)\.weight$", name):
            return 2
        return 1

    def _iter_ipc_weight_chunks(self, policy_model):
        target_device_indices = self._get_ipc_target_device_indices()
        max_chunk_bytes = int(self.ipc_chunk_mb * 1024 * 1024)
        chunk_names: List[str] = []
        chunk_dtype_names: List[str] = []
        chunk_shapes: List[List[int]] = []
        chunk_ipc_handles: List[Dict[str, Tuple[Any, tuple]]] = []
        chunk_refs: List[torch.Tensor] = []
        chunk_bytes = 0
        pending_group_key: Optional[str] = None
        pending_group_expected = 0
        pending_group_count = 0
        deferred_copy_error: Optional[str] = None

        def _flush_chunk():
            nonlocal chunk_names, chunk_dtype_names, chunk_shapes, chunk_ipc_handles, chunk_refs, chunk_bytes
            if not chunk_names:
                return None
            update_info = {
                "names": chunk_names,
                "dtype_names": chunk_dtype_names,
                "shapes": chunk_shapes,
                "ipc_handles": chunk_ipc_handles,
            }
            refs = chunk_refs
            chunk_names = []
            chunk_dtype_names = []
            chunk_shapes = []
            chunk_ipc_handles = []
            chunk_refs = []
            chunk_bytes = 0
            return update_info, refs

        def _can_flush_between_groups() -> bool:
            return pending_group_key is None or pending_group_count >= pending_group_expected

        def _mark_group_progress(name: str) -> None:
            nonlocal pending_group_key, pending_group_expected, pending_group_count
            group_key = self._get_ipc_atomic_group_key(name)
            if group_key is None:
                pending_group_key = None
                pending_group_expected = 0
                pending_group_count = 0
                return

            expected = self._get_ipc_atomic_group_expected_size(name)
            if pending_group_key != group_key:
                pending_group_key = group_key
                pending_group_expected = expected
                pending_group_count = 0

            pending_group_count += 1
            if pending_group_count >= pending_group_expected:
                pending_group_key = None
                pending_group_expected = 0
                pending_group_count = 0

        for name, p in self._iter_sync_named_parameters(policy_model):
            try:
                with self._zero_gather_context(p):
                    if self.shared_vllm and self.rank != self.engine_owner_rank:
                        continue
                    # After a copy error, keep entering every remaining ZeRO
                    # gather so peers do not block, but do no more CUDA work.
                    if deferred_copy_error is not None:
                        continue

                    full_weight = p.detach()
                    full_shape = [int(dim) for dim in full_weight.shape]
                    dtype_name = str(full_weight.dtype).split(".")[-1]
                    tensor_bytes = max(1, int(full_weight.numel()) * int(full_weight.element_size()))
                    per_device_handles: Dict[str, Tuple[Any, tuple]] = {}
                    parameter_refs: List[torch.Tensor] = []

                    for device_index in target_device_indices:
                        copied = full_weight.to(
                            device=torch.device("cuda", device_index),
                            copy=True,
                        ).contiguous()
                        torch.cuda.synchronize(device_index)
                        per_device_handles[self._get_cuda_device_uuid(device_index)] = reduce_tensor(copied)
                        parameter_refs.append(copied)

                    chunk_refs.extend(parameter_refs)
                    chunk_names.append(name)
                    chunk_dtype_names.append(dtype_name)
                    chunk_shapes.append(full_shape)
                    chunk_ipc_handles.append(per_device_handles)
                    chunk_bytes += tensor_bytes
                    _mark_group_progress(name)
            except Exception as exc:
                deferred_copy_error = (
                    f"parameter={name} {type(exc).__name__}: {exc}\n"
                    + traceback.format_exc(limit=20)
                )
                # A sync with a missing parameter is unusable. Drop any
                # not-yet-sent chunk while retaining collective alignment.
                chunk_names = []
                chunk_dtype_names = []
                chunk_shapes = []
                chunk_ipc_handles = []
                chunk_refs = []
                chunk_bytes = 0
                pending_group_key = None
                pending_group_expected = 0
                pending_group_count = 0
                continue

            if (
                deferred_copy_error is None
                and chunk_bytes >= max_chunk_bytes
                and _can_flush_between_groups()
            ):
                flushed = _flush_chunk()
                if flushed is not None:
                    yield flushed

        if deferred_copy_error is not None:
            raise RuntimeError("IPC CUDA copy failed after completing ZeRO gathers:\n" + deferred_copy_error)
        flushed = _flush_chunk()
        if flushed is not None:
            yield flushed

    def _reset_prefix_cache_after_weight_sync(self) -> None:
        """Invalidate KV blocks produced by the previous policy version."""
        reset_ok = self.llm.reset_prefix_cache()
        if not bool(reset_ok):
            raise RuntimeError(
                "vLLM refused to reset its prefix cache after the policy weight sync."
            )
        print(
            "[UserRolloutBackend] reset vLLM prefix cache after policy weight sync",
            flush=True,
        )

    def _ipc_sync_llm_from_policy(self, policy_model) -> bool:
        self._ensure_runtime()
        if not self._maybe_init_ipc_weight_transfer_engine():
            return False

        sync_error = None
        try:
            target_devices = self._get_ipc_target_device_indices()
            print(
                "[UserRolloutBackend] syncing rollout weights via ipc "
                f"(target_devices={target_devices}, chunk_mb={self.ipc_chunk_mb:.1f})",
                flush=True,
            )
            for update_info, refs in self._iter_ipc_weight_chunks(policy_model):
                if sync_error is None:
                    try:
                        self.llm.update_weights(dict(update_info=update_info))
                    except Exception as exc:
                        sync_error = (
                            f"{type(exc).__name__}: {exc}\n"
                            + traceback.format_exc(limit=20)
                        )
                refs.clear()
            if sync_error is None:
                self._reset_prefix_cache_after_weight_sync()
        except Exception as exc:
            sync_error = (
                f"{type(exc).__name__}: {exc}\n"
                + traceback.format_exc(limit=20)
            )
        if sync_error is not None:
            print(
                f"[UserRolloutBackend] IPC weight sync/cache reset failed:\n{sync_error}",
                flush=True,
            )
            return False
        return True

    def _ipc_sync_llm_from_policy_shared(self, policy_model) -> bool:
        if not _dist_is_initialized():
            return False

        ok = True
        if self.rank == self.engine_owner_rank:
            self._ensure_runtime()
            try:
                self._maybe_init_ipc_weight_transfer_engine()
            except Exception as exc:
                print(f"[UserRolloutBackend] shared IPC init failed: {exc}", flush=True)
                ok = False

        ok_payload = [ok if self.rank == self.engine_owner_rank else None]
        dist.broadcast_object_list(ok_payload, src=self.engine_owner_rank)
        ok = bool(ok_payload[0])
        dist.barrier()
        if not ok:
            return False

        if self.rank == self.engine_owner_rank:
            sync_error = None
            try:
                target_devices = self._get_ipc_target_device_indices()
                print(
                    "[UserRolloutBackend] syncing rollout weights via ipc "
                    f"(target_devices={target_devices}, chunk_mb={self.ipc_chunk_mb:.1f})",
                    flush=True,
                )
                # Even if one vLLM update fails, consume the remaining generator
                # so every rank enters the same ZeRO GatheredParameters sequence.
                # The partially updated engine is rejected after collectives end.
                for update_info, refs in self._iter_ipc_weight_chunks(policy_model):
                    if sync_error is None:
                        try:
                            self.llm.update_weights(dict(update_info=update_info))
                        except Exception as exc:
                            sync_error = (
                                f"{type(exc).__name__}: {exc}\n"
                                + traceback.format_exc(limit=20)
                            )
                    refs.clear()
                if sync_error is None:
                    self._reset_prefix_cache_after_weight_sync()
            except Exception as exc:
                sync_error = (
                    f"{type(exc).__name__}: {exc}\n"
                    + traceback.format_exc(limit=20)
                )
            if sync_error is not None:
                print(
                    f"[UserRolloutBackend] shared IPC weight sync/cache reset failed:\n{sync_error}",
                    flush=True,
                )
                ok = False
        else:
            try:
                self._participate_in_zero_gather_for_sync(policy_model)
            except Exception as exc:
                print(f"[UserRolloutBackend] shared ZeRO gather sync failed on rank={self.rank}: {exc}", flush=True)
                ok = False

        # Include peer-side gather failures as well; an owner-only broadcast
        # would overwrite a peer's local ``ok=False`` state.
        ok_tensor = torch.tensor(
            1 if ok else 0,
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        dist.barrier()
        return bool(ok_tensor.item())

    def maybe_sync_from_policy_model(self, policy_model, step: int):
        if policy_model is None:
            return
        if step < 0:
            return
        # The bootstrap sync is unconditional.  The interval applies only
        # after vLLM has received at least one policy snapshot.
        if self._last_synced_step >= 0 and (step % self.sync_interval_steps) != 0:
            return
        if step == self._last_synced_step:
            return

        try:
            self._prepare_zero_for_sync(policy_model)
        except Exception as exc:
            raise RuntimeError(
                "Failed to prepare DeepSpeed ZeRO parameters for rollout sync: "
                f"{type(exc).__name__}: {exc!r}"
            ) from exc

        if self.shared_vllm:
            sync_ok = self._ipc_sync_llm_from_policy_shared(policy_model)
        else:
            sync_ok = self._ipc_sync_llm_from_policy(policy_model)

        if not sync_ok:
            raise RuntimeError(
                f"{self.weight_transfer_backend.upper()} sync failed at step={step}; "
                "fallback paths are disabled by design."
            )

        self._last_synced_step = int(step)
        if (not self.shared_vllm) or self.rank == self.engine_owner_rank:
            print(
                f"[UserRolloutBackend] synced rollout weights from policy model at step={step} "
                f"({self.weight_transfer_backend})",
                flush=True,
            )

    def _prepare_expanded_items(
        self, requests: List[RolloutRequest]
    ) -> Tuple[List[Tuple[int, int, list, Dict[str, Any]]], List[Optional[str]]]:
        expanded_items: List[Tuple[int, int, list, Dict[str, Any]]] = []
        anchor_images: List[Optional[str]] = []
        vision_model = self._get_rollout_vision_model()
        if requests and vision_model is None:
            raise RuntimeError(
                "Rollout vision model is unavailable on this rank. "
                "With owner-only shared vision enabled, only owner_rank should precompute mm_data."
            )
        for request_idx, request in enumerate(requests):
            anchor_image = request.anchor_image or select_anchor_image_path(request.messages)
            anchor_images.append(anchor_image)
            base_messages = request.messages
            base_input = prepare_inputs_for_vllm_external(
                base_messages,
                self.processor,
                vision_model,
                image_branch_flags=request.image_branch_flags,
            )
            base_prompt = base_input["prompt"]
            base_prompt_token_ids = list(base_input["prompt_token_ids"])
            base_mm_data = base_input.get("multi_modal_data", {}) or {}
            base_image_branch_flags = list(base_input.get("image_branch_flags", []))
            for gen_idx in range(max(1, int(request.num_generations))):
                message_copy = _clone_rollout_messages(base_messages)
                one_input = {
                    "prompt": base_prompt,
                    "prompt_token_ids": list(base_prompt_token_ids),
                    # Generation states replace mm_data via copy-on-write after pause updates,
                    # so the immutable base mm_data can be shared across generations.
                    "multi_modal_data": base_mm_data,
                    "image_branch_flags": list(base_image_branch_flags),
                    "anchor_image": anchor_image,
                }
                expanded_items.append((request_idx, gen_idx, message_copy, one_input))
        return expanded_items, anchor_images

    @staticmethod
    def _flatten_prepared_per_rank(
        gathered_requests: List[Optional[List[RolloutRequest]]],
        prepared_per_rank: List[Tuple[List[Tuple[int, int, list, Dict[str, Any]]], List[Optional[str]]]],
    ) -> Tuple[
        List[RolloutRequest],
        List[Tuple[int, int, list, Dict[str, Any]]],
        List[Optional[str]],
        List[Tuple[int, int]],
        List[int],
    ]:
        flat_requests: List[RolloutRequest] = []
        flat_expanded_items: List[Tuple[int, int, list, Dict[str, Any]]] = []
        flat_anchor_images: List[Optional[str]] = []
        per_rank_request_slices: List[Tuple[int, int]] = []
        flat_expanded_item_owner_ranks: List[int] = []

        request_offset = 0
        for rank_idx, rank_requests in enumerate(gathered_requests):
            rank_requests = rank_requests or []
            rank_expanded_items, rank_anchor_images = (
                prepared_per_rank[rank_idx] if rank_idx < len(prepared_per_rank) else ([], [])
            )

            per_rank_request_slices.append((len(flat_requests), len(rank_requests)))
            flat_requests.extend(rank_requests)
            flat_anchor_images.extend(rank_anchor_images)
            for request_idx, gen_idx, message_copy, one_input in rank_expanded_items:
                flat_expanded_items.append((request_offset + request_idx, gen_idx, message_copy, one_input))
                flat_expanded_item_owner_ranks.append(rank_idx)
            request_offset += len(rank_requests)

        return (
            flat_requests,
            flat_expanded_items,
            flat_anchor_images,
            per_rank_request_slices,
            flat_expanded_item_owner_ranks,
        )

    def _generate_from_expanded(
        self,
        requests: List[RolloutRequest],
        expanded_items: List[Tuple[int, int, list, Dict[str, Any]]],
        anchor_images: List[Optional[str]],
        shared_generate: bool = False,
        shared_vision_owner_only: Optional[bool] = None,
        expanded_item_owner_ranks: Optional[List[int]] = None,
    ) -> List[RolloutResponse]:
        if not expanded_items:
            return [
                RolloutResponse(
                    uid=request.uid,
                    completions=[],
                    anchor_image=anchor_images[idx] if idx < len(anchor_images) else request.anchor_image,
                )
                for idx, request in enumerate(requests)
            ]
        if expanded_item_owner_ranks is not None and len(expanded_item_owner_ranks) != len(expanded_items):
            raise ValueError(
                "expanded_item_owner_ranks length must match expanded_items length: "
                f"{len(expanded_item_owner_ranks)} != {len(expanded_items)}."
            )

        sample_request = requests[0]
        base_params = self._build_sampling_params(sample_request)
        inputs = [item[3] for item in expanded_items]
        messages = [item[2] for item in expanded_items]
        report_list = [{"question_id": idx} for idx in range(len(expanded_items))]

        results = iterative_generate_dynamic_batch_external(
            llm=self.llm,
            processor=self.processor,
            vision_model=self.vision_model,
            inputs=inputs,
            messages=messages,
            report_list=report_list,
            base_params=base_params,
            pause_token_id=self.pause_token_id,
            true_stop_token_id=self._get_true_stop_token_ids(),
            max_pause_times=self.max_pause_times,
            max_total_new_tokens=sample_request.max_new_tokens,
            shared_generate_owner_rank=self.engine_owner_rank if shared_generate else None,
            shared_generate_rank=self.rank,
            shared_generate_group=self.rollout_control_group if shared_generate else None,
            shared_vision_owner_only=(
                self.shared_vision_owner_only if shared_vision_owner_only is None else shared_vision_owner_only
            )
            and shared_generate,
            state_owner_ranks=expanded_item_owner_ranks,
            max_model_len=self.vllm_max_model_len,
        )

        completions_by_request_idx: Dict[int, Dict[int, Completion]] = {idx: {} for idx in range(len(requests))}
        for local_idx, result in enumerate(results):
            request_idx, gen_idx, _, _ = expanded_items[local_idx]
            completions_by_request_idx[request_idx][gen_idx] = Completion(
                text=result.get("text", ""),
                token_ids=list(result.get("token_ids", [])),
                action_mask=list(result.get("action_mask", [])),
            )

        responses: List[RolloutResponse] = []
        for request_idx, request in enumerate(requests):
            ordered: List[Completion] = []
            slot_map = completions_by_request_idx.get(request_idx, {})
            for gen_idx in range(max(1, int(request.num_generations))):
                ordered.append(slot_map.get(gen_idx, Completion(text="")))
            responses.append(
                RolloutResponse(
                    uid=request.uid,
                    completions=ordered,
                    anchor_image=anchor_images[request_idx] if request_idx < len(anchor_images) else request.anchor_image,
                )
            )
        return responses

    def _generate_local(self, requests: List[RolloutRequest]) -> List[RolloutResponse]:
        self._ensure_runtime()
        expanded_items, anchor_images = self._prepare_expanded_items(requests)
        return self._generate_from_expanded(requests, expanded_items, anchor_images)

    def _generate_shared(self, requests: List[RolloutRequest]) -> List[RolloutResponse]:
        gathered_requests: List[Optional[List[RolloutRequest]]] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_requests, requests, group=self.rollout_control_group)

        self._ensure_runtime()
        if self.shared_prepare_all_ranks:
            with self._shared_visual_gather_context():
                local_prepared = self._prepare_expanded_items(requests)

            gathered_prepared: List[Optional[Tuple[List[Tuple[int, int, list, Dict[str, Any]]], List[Optional[str]]]]] = [
                None for _ in range(self.world_size)
            ]
            dist.all_gather_object(
                gathered_prepared,
                local_prepared,
                group=self.rollout_control_group,
            )
            prepared_per_rank = [payload if payload is not None else ([], []) for payload in gathered_prepared]

            (
                flat_requests,
                flat_expanded_items,
                flat_anchor_images,
                per_rank_request_slices,
                flat_expanded_item_owner_ranks,
            ) = self._flatten_prepared_per_rank(gathered_requests, prepared_per_rank)
            all_rank_responses = self._generate_from_expanded(
                flat_requests,
                flat_expanded_items,
                flat_anchor_images,
                shared_generate=True,
                shared_vision_owner_only=False,
                expanded_item_owner_ranks=flat_expanded_item_owner_ranks,
            )
            start, count = per_rank_request_slices[self.rank]
            return all_rank_responses[start : start + count]

        prepared_per_rank: List[
            Tuple[List[Tuple[int, int, list, Dict[str, Any]]], List[Optional[str]]]
        ] = []
        response_payload = [None]
        if self.shared_vision_owner_only:
            prepared_payload = [None]
            if self.rank == self.engine_owner_rank:
                owner_prepared_per_rank = None
                prepare_error = None
                try:
                    with self._shared_visual_gather_context():
                        owner_prepared_per_rank = []
                        for rank_idx, rank_requests in enumerate(gathered_requests):
                            rank_requests = rank_requests or []
                            owner_prepared_per_rank.append(self._prepare_expanded_items(rank_requests))
                except Exception as exc:
                    prepare_error = (
                        f"{type(exc).__name__}: {exc}\n"
                        + traceback.format_exc(limit=20)
                    )
                prepared_payload[0] = {
                    "payload": owner_prepared_per_rank,
                    "error": prepare_error,
                }
            dist.broadcast_object_list(
                prepared_payload,
                src=self.engine_owner_rank,
                group=self.rollout_control_group,
            )
            shared_prepared = prepared_payload[0]
            if not isinstance(shared_prepared, dict):
                raise RuntimeError("Shared rollout preparation broadcast returned an invalid envelope.")
            if shared_prepared.get("error"):
                raise RuntimeError(
                    "Shared rollout preparation failed on the owner rank:\n"
                    + str(shared_prepared["error"])
                )
            prepared_per_rank = shared_prepared.get("payload")
            if prepared_per_rank is None:
                raise RuntimeError("Shared rollout preparation broadcast returned no payload.")
        else:
            with self._shared_visual_gather_context():
                for rank_idx, rank_requests in enumerate(gathered_requests):
                    rank_requests = rank_requests or []
                    prepared_per_rank.append(self._prepare_expanded_items(rank_requests))

        if self.rank == self.engine_owner_rank:
            per_rank_responses: List[List[RolloutResponse]] = []
            for rank_idx, rank_requests in enumerate(gathered_requests):
                rank_requests = rank_requests or []
                expanded_items, anchor_images = prepared_per_rank[rank_idx]
                per_rank_responses.append(
                    self._generate_from_expanded(
                        rank_requests,
                        expanded_items,
                        anchor_images,
                        shared_generate=True,
                    )
                )
            response_payload[0] = per_rank_responses
        else:
            for rank_idx, rank_requests in enumerate(gathered_requests):
                rank_requests = rank_requests or []
                expanded_items, anchor_images = prepared_per_rank[rank_idx]
                self._generate_from_expanded(
                    rank_requests,
                    expanded_items,
                    anchor_images,
                    shared_generate=True,
                )
        dist.barrier(group=self.rollout_control_group)

        dist.broadcast_object_list(
            response_payload,
            src=self.engine_owner_rank,
            group=self.rollout_control_group,
        )
        all_rank_responses = response_payload[0]
        if all_rank_responses is None:
            raise RuntimeError("Shared vLLM generate broadcast returned no responses.")
        return all_rank_responses[self.rank]

    def generate(self, requests: List[RolloutRequest]) -> List[RolloutResponse]:
        if not requests and not self.shared_vllm:
            return []

        if self.shared_vllm:
            return self._generate_shared(requests)
        return self._generate_local(requests)
