from __future__ import annotations

import contextlib
import os
import importlib
import re
import traceback
import warnings
from typing import Any, List, Callable, Dict, Optional, Tuple
import numpy as np
import torch
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from qwenvl.rl_grpo_verl.chexbert import CheXbertLabeler, ReasoningCheXbertLabeler

_ANSWER_CHEXBERT_SCORER = None
_ANSWER_CHEXBERT_IMPORT_ERROR: Optional[Exception] = None
_ANSWER_CHEXBERT_LABEL_CACHE: Dict[str, list[int]] = {}
_REASONING_CHEXBERT_SCORER = None
_REASONING_CHEXBERT_IMPORT_ERROR: Optional[Exception] = None
_REASONING_CHEXBERT_LABEL_CACHE: Dict[str, list[int]] = {}
_RADGRAPH_SCORER = None
_RADGRAPH_IMPORT_ERROR: Optional[Exception] = None
_RADGRAPH_SCORE_CACHE: Dict[Tuple[str, str], float] = {}
_RADGRAPH_MODEL_CACHE_DIR = os.environ.get(
    "QWENVL_RADGRAPH_MODEL_CACHE_DIR",
    os.path.expanduser("~/.cache/mcot/radgraph"),
)
_RADGRAPH_CACHED_TRANSFORMERS_PATCHED = False
_REWARD_BACKENDS_PREWARMED = False
_OBJECT_REF_BOX_PATTERN = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
    r"<\|box_start\|>(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)<\|box_end\|>",
    re.DOTALL,
)
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_BOX_IMAGE_BLOCK_PATTERN = re.compile(
    r"[ \t]*(?:<\|object_ref_start\|>.*?<\|object_ref_end\|>)?"
    r"<\|box_start\|>.*?<\|box_end\|>[ \t]*(?:\r?\n)?[ \t]*(?:<image>[ \t]*(?:\r?\n)?[ \t]*|$)",
    re.DOTALL,
)
_RESIDUAL_OBJECT_REF_LINE_PATTERN = re.compile(
    r"^[ \t]*<\|object_ref_start\|>[^\n\r]*(?:\r?\n)?",
    re.MULTILINE,
)
_RESIDUAL_BOX_LINE_PATTERN = re.compile(
    r"^[ \t]*<\|box_(?:start|end)\|>[^\n\r]*(?:\r?\n)?",
    re.MULTILINE,
)
_RESIDUAL_IMAGE_LINE_PATTERN = re.compile(
    r"^[ \t]*<image>[ \t]*(?:\r?\n)?",
    re.MULTILINE,
)
_IMAGE_TAG_WITH_SURROUNDING_WHITESPACE_PATTERN = re.compile(
    r"[ \t]*(?:\r?\n)?[ \t]*<image>[ \t]*(?:\r?\n)?[ \t]*"
)


def _default_reward_device_specs() -> Tuple[str, str, str]:
    if not torch.cuda.is_available():
        return ("cpu", "cpu", "cpu")
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 3:
        return ("cuda:0", "cuda:1", "cuda:2")
    if gpu_count == 2:
        return ("cuda:0", "cuda:1", "cuda:1")
    return ("cuda:0", "cuda:0", "cuda:0")


_DEFAULT_ANSWER_DEVICE, _DEFAULT_REASONING_DEVICE, _DEFAULT_RADGRAPH_DEVICE = _default_reward_device_specs()
_ANSWER_CHEXBERT_DEVICE = os.environ.get("QWENVL_ANSWER_CHEXBERT_DEVICE", _DEFAULT_ANSWER_DEVICE)
_REASONING_CHEXBERT_DEVICE = os.environ.get("QWENVL_REASONING_CHEXBERT_DEVICE", _DEFAULT_REASONING_DEVICE)
_RADGRAPH_DEVICE = os.environ.get("QWENVL_RADGRAPH_DEVICE", _DEFAULT_RADGRAPH_DEVICE)
_ANSWER_CHEXBERT_BATCH_SIZE = int(os.environ.get("QWENVL_ANSWER_CHEXBERT_BATCH_SIZE", "16"))
_REASONING_CHEXBERT_BATCH_SIZE = int(os.environ.get("QWENVL_REASONING_CHEXBERT_BATCH_SIZE", "4"))
_RADGRAPH_BATCH_SIZE = int(os.environ.get("QWENVL_RADGRAPH_BATCH_SIZE", "8"))
_REWARD_BATCH_SIZE = int(os.environ.get("QWENVL_REWARD_BATCH_SIZE", "20"))
_BBOX_FP_PENALTY_LAMBDA = 0.05


def _env_flag(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default)
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


_ENABLE_REASONING_CHEXBERT = _env_flag("QWENVL_ENABLE_REASONING_CHEXBERT", "1")
_ENABLE_RADGRAPH = _env_flag("QWENVL_ENABLE_RADGRAPH", "1")


def _radgraph_cuda_arg(device_spec: str) -> int:
    normalized = str(device_spec).strip().lower()
    if normalized == "cpu":
        return -1
    if normalized.startswith("cuda:"):
        normalized = normalized.split(":", 1)[1]
    return int(normalized)


def _radgraph_torch_device() -> torch.device:
    normalized = str(_RADGRAPH_DEVICE).strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    return torch.device(normalized)


def _empty_radgraph_cuda_cache() -> None:
    device = _radgraph_torch_device()
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    except Exception:
        pass


def _mean_radgraph_f1(values: Any) -> float:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().flatten().tolist()
    elif isinstance(values, np.ndarray):
        values = values.reshape(-1).tolist()
    if isinstance(values, (list, tuple)):
        if not values:
            return 0.0
        return float(sum(float(value) for value in values)) / float(len(values))
    return float(values)


def _parse_radgraph_batch_output(radgraph_output: Any, expected_count: int) -> List[float]:
    if not isinstance(radgraph_output, (list, tuple)) or len(radgraph_output) < 2:
        raise ValueError(f"Unexpected F1RadGraph output: {radgraph_output!r}")

    reward_list = radgraph_output[1]
    if not isinstance(reward_list, (list, tuple)):
        raise ValueError(f"Unexpected F1RadGraph reward list: {radgraph_output!r}")

    # F1RadGraph(reward_level='all') returns:
    # (mean_reward, ([simple...], [partial...], [complete...]), hyp_annos, ref_annos)
    if (
        len(reward_list) == 3
        and all(isinstance(component, (list, tuple, np.ndarray)) for component in reward_list)
        and all(len(component) == expected_count for component in reward_list)
    ):
        return [
            _mean_radgraph_f1([reward_list[0][idx], reward_list[1][idx], reward_list[2][idx]])
            for idx in range(expected_count)
        ]

    if len(reward_list) == expected_count:
        return [_mean_radgraph_f1(one_score) for one_score in reward_list]

    raise ValueError(f"Unexpected F1RadGraph reward count: {radgraph_output!r}")


def _run_radgraph_scorer(scorer: Any, refs: List[str], hyps: List[str]) -> Any:
    with torch.inference_mode():
        output = scorer(refs=refs, hyps=hyps)
    _empty_radgraph_cuda_cache()
    return output


def _current_dist_runtime_context() -> str:
    rank = 0
    world_size = 1
    dist_initialized = False
    try:
        import torch.distributed as dist

        dist_initialized = bool(dist.is_available() and dist.is_initialized())
        if dist_initialized:
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
    except Exception:
        pass

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = 0
    current_cuda_device: Optional[int] = None
    try:
        cuda_device_count = int(torch.cuda.device_count())
        if cuda_available:
            current_cuda_device = int(torch.cuda.current_device())
    except Exception:
        pass

    return (
        f"pid={os.getpid()} rank={rank} world_size={world_size} "
        f"dist_initialized={dist_initialized} cuda_available={cuda_available} "
        f"cuda_device_count={cuda_device_count} current_cuda_device={current_cuda_device} "
        f"configured_radgraph_device={_RADGRAPH_DEVICE}"
    )


def _reward_log(message: str) -> None:
    print(f"[reward][debug] {message}", flush=True)


def _iter_module_subclasses(base_cls: type[torch.nn.Module]):
    seen: set[type[torch.nn.Module]] = set()
    stack = list(base_cls.__subclasses__())
    while stack:
        subclass = stack.pop()
        if subclass in seen:
            continue
        seen.add(subclass)
        yield subclass
        stack.extend(subclass.__subclasses__())


def _is_deepspeed_wrapped_init(module_cls: type[torch.nn.Module]) -> bool:
    old_init = getattr(module_cls, "_old_init", None)
    if old_init is None:
        return False
    return getattr(module_cls, "__init__", None) is not old_init


def _force_restore_deepspeed_wrapped_module_inits() -> int:
    restored = 0
    for module_cls in _iter_module_subclasses(torch.nn.Module):
        old_init = getattr(module_cls, "_old_init", None)
        if old_init is None:
            continue
        current_init = getattr(module_cls, "__init__", None)
        if current_init is old_init:
            continue
        module_cls.__init__ = old_init
        restored += 1
    return restored


def _patch_radgraph_cached_transformers_get() -> None:
    global _RADGRAPH_CACHED_TRANSFORMERS_PATCHED
    if _RADGRAPH_CACHED_TRANSFORMERS_PATCHED:
        return

    import copy
    import radgraph.allennlp.common.cached_transformers as cached_transformers
    from transformers.models.bert.modeling_bert import BertModel

    original_get = cached_transformers.get

    def _ds_safe_get(
        model_name: str,
        make_copy: bool,
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
    ):
        if override_weights_file is not None:
            return original_get(
                model_name,
                make_copy,
                override_weights_file,
                override_weights_strip_prefix,
            )

        spec = cached_transformers.TransformerSpec(
            model_name, override_weights_file, override_weights_strip_prefix
        )
        transformer = cached_transformers._model_cache.get(spec, None)
        if transformer is None:
            config = cached_transformers.AutoConfig.from_pretrained(model_name)
            if getattr(config, "model_type", None) == "bert":
                _reward_log(
                    f"radgraph cached_transformers: constructing BertModel(config) for {model_name}"
                )
                transformer = BertModel(config)
            else:
                transformer = cached_transformers.AutoModel.from_config(config)
            cached_transformers._model_cache[spec] = transformer

        if make_copy:
            return copy.deepcopy(transformer)
        return transformer

    cached_transformers.get = _ds_safe_get
    _RADGRAPH_CACHED_TRANSFORMERS_PATCHED = True
    _reward_log("radgraph cached_transformers.get monkey patch installed")


@contextlib.contextmanager
def _temporarily_disable_deepspeed_zero_init(scope: str = "unknown"):
    try:
        import deepspeed
    except Exception:
        _reward_log(f"{scope}: deepspeed import unavailable; zero-init guard bypassed")
        yield
        return

    partition_parameters = getattr(getattr(deepspeed, "zero", None), "partition_parameters", None)
    if partition_parameters is None:
        _reward_log(f"{scope}: deepspeed.zero.partition_parameters unavailable; zero-init guard bypassed")
        yield
        return

    shutdown_init_context = getattr(partition_parameters, "shutdown_init_context", None)
    restore_init_context = getattr(partition_parameters, "restore_init_context", None)
    if shutdown_init_context is None or restore_init_context is None:
        _reward_log(f"{scope}: zero-init shutdown/restore hooks unavailable; guard bypassed")
        yield
        return

    _reward_log(f"{scope}: disabling deepspeed zero.Init wrappers")
    shutdown_init_context()
    try:
        bert_wrapped_after_shutdown = None
        try:
            from transformers.models.bert.modeling_bert import BertModel

            bert_wrapped_after_shutdown = _is_deepspeed_wrapped_init(BertModel)
        except Exception as exc:
            _reward_log(f"{scope}: unable to inspect BertModel init wrapper state: {exc}")

        if bert_wrapped_after_shutdown:
            restored_count = _force_restore_deepspeed_wrapped_module_inits()
            _reward_log(
                f"{scope}: BertModel remained wrapped after shutdown_init_context; "
                f"manually restored {restored_count} module __init__ wrappers"
            )
        elif bert_wrapped_after_shutdown is False:
            _reward_log(f"{scope}: BertModel init wrapper cleared after shutdown_init_context")

        _reward_log(f"{scope}: zero.Init wrappers disabled")
        yield
    finally:
        _reward_log(f"{scope}: restoring deepspeed zero.Init wrappers")
        restore_init_context()
        _reward_log(f"{scope}: zero.Init wrappers restored")

def bbox_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union



# def bbox_reward_with_fp_penalty(
#     gt_bbox_dict: Dict[str, List[float]],
#     pred_bbox_dict: Dict[str, List[float]],
#     fp_penalty_lambda: float = _BBOX_FP_PENALTY_LAMBDA,
# ) -> float:
#     if not gt_bbox_dict:
#         extra_pred_count = len(pred_bbox_dict)
#         return float(-fp_penalty_lambda * extra_pred_count)

#     total_iou = 0.0
#     for key, gt_bbox in gt_bbox_dict.items():
#         pred_bbox = pred_bbox_dict.get(key)
#         if pred_bbox is None:
#             continue
#         total_iou += bbox_iou(gt_bbox, pred_bbox)

#     avg_iou = total_iou / len(gt_bbox_dict)
#     extra_pred_count = len(set(pred_bbox_dict) - set(gt_bbox_dict))
#     return float(avg_iou - fp_penalty_lambda * extra_pred_count)

def bbox_reward_with_fp_penalty(
    gt_bbox_dict: Dict[str, List[float]],
    pred_bbox_dict: Dict[str, List[float]],
    fp_penalty_lambda: float = _BBOX_FP_PENALTY_LAMBDA,
) -> float:
    """
    Compute weighted bbox IoU reward with false-positive penalty.

    Region weights are computed internally from region occurrence counts:

        p_c = count_c / sum_j count_j
        raw_weight_c = 1 / p_c^alpha
        weight_c = raw_weight_c / mean_j(raw_weight_j)

    Main behavior:
    - GT regions are weighted by their inverse-frequency weights.
    - Missing GT regions contribute 0 IoU but still contribute to the denominator.
    - Extra predicted regions are penalized.
    - By default, FP penalty is based on the number of extra predictions.
    
    mcot_region_to_count = {'right lung': 152734, 'right upper lung zone': 4757, 'right mid lung zone': 7654, 'right lower lung zone': 31442, 'right hilar structures': 48140, 'right apical zone': 4583, 'right costophrenic angle': 88237, 'right hemidiaphragm': 8905, 
    'left lung': 147063, 'left upper lung zone': 4204, 'left mid lung zone': 1880, 'left lower lung zone': 33104, 'left hilar structures': 5970, 'left apical zone': 2847, 'left costophrenic angle': 28197, 'left hemidiaphragm': 3157, 'trachea': 17302, 'spine': 26426, 
    'right clavicle': 6765, 'left clavicle': 1534, 'aortic arch': 8939, 'mediastinum': 113222, 'upper mediastinum': 2820, 'svc': 8576, 'cardiac silhouette': 119304, 'cavoatrial junction': 2578, 'right atrium': 4427, 'carina': 6552, 'abdomen': 25310}

    abn_mcot_region_to_count = {'right lung': 33492, 'right upper lung zone': 1015, 'right mid lung zone': 2072, 'right lower lung zone': 6706, 'right hilar structures': 11108, 'right apical zone': 951, 'right costophrenic angle': 21559, 'right hemidiaphragm': 2577, 
    'left lung': 32656, 'left upper lung zone': 673, 'left mid lung zone': 476, 'left lower lung zone': 6928, 'left hilar structures': 1268, 'left apical zone': 613, 'left costophrenic angle': 5135, 'left hemidiaphragm': 644, 'trachea': 645, 'spine': 9381, 
    'right clavicle': 1559, 'left clavicle': 198, 'aortic arch': 2815, 'mediastinum': 26822, 'upper mediastinum': 373, 'svc': 1060, 'cardiac silhouette': 29285, 'cavoatrial junction': 334, 'right atrium': 946, 'carina': 13, 'abdomen': 3348}
    """
    
    alpha = 0.25
    default_weight = 1.0

    # mcot_region_to_count = {'right lung': 152734, 'right upper lung zone': 4757, 'right mid lung zone': 7654, 'right lower lung zone': 31442, 'right hilar structures': 48140, 'right apical zone': 4583, 'right costophrenic angle': 88237, 'right hemidiaphragm': 8905, 
    # 'left lung': 147063, 'left upper lung zone': 4204, 'left mid lung zone': 1880, 'left lower lung zone': 33104, 'left hilar structures': 5970, 'left apical zone': 2847, 'left costophrenic angle': 28197, 'left hemidiaphragm': 3157, 'trachea': 17302, 'spine': 26426, 
    # 'right clavicle': 6765, 'left clavicle': 1534, 'aortic arch': 8939, 'mediastinum': 113222, 'upper mediastinum': 2820, 'svc': 8576, 'cardiac silhouette': 119304, 'cavoatrial junction': 2578, 'right atrium': 4427, 'carina': 6552, 'abdomen': 25310}

    mcot_region_to_count = {'right lung': 33492, 'right upper lung zone': 1015, 'right mid lung zone': 2072, 'right lower lung zone': 6706, 'right hilar structures': 11108, 'right apical zone': 951, 'right costophrenic angle': 21559, 'right hemidiaphragm': 2577, 
    'left lung': 32656, 'left upper lung zone': 673, 'left mid lung zone': 476, 'left lower lung zone': 6928, 'left hilar structures': 1268, 'left apical zone': 613, 'left costophrenic angle': 5135, 'left hemidiaphragm': 644, 'trachea': 645, 'spine': 9381, 
    'right clavicle': 1559, 'left clavicle': 198, 'aortic arch': 2815, 'mediastinum': 26822, 'upper mediastinum': 373, 'svc': 1060, 'cardiac silhouette': 29285, 'cavoatrial junction': 334, 'right atrium': 946, 'carina': 13, 'abdomen': 3348}

    region_to_count = mcot_region_to_count

    # Build region weights.
    total_count = sum(region_to_count.values())

    raw_weights = {}
    for region, count in region_to_count.items():
        p_c = count / total_count
        raw_weights[region] = 1.0 / ((p_c) ** alpha)

    mean_raw_weight = sum(raw_weights.values()) / len(raw_weights)

    region_weights = {
        region: raw_weight / (mean_raw_weight)
        for region, raw_weight in raw_weights.items()
    }

    # Case 1: no GT bbox.
    # Only penalize hallucinated predicted bboxes.
    if not gt_bbox_dict:
        extra_pred_regions = set(pred_bbox_dict)
        fp_penalty = len(extra_pred_regions)

        return float(-fp_penalty_lambda * fp_penalty)

    # Case 2: normal case with GT bboxes.
    weighted_iou_sum = 0.0
    weight_sum = 0.0

    for region, gt_bbox in gt_bbox_dict.items():
        weight = region_weights.get(region, default_weight)
        weight_sum += weight

        pred_bbox = pred_bbox_dict.get(region)

        # Missing GT region contributes 0 IoU but is still counted in denominator.
        if pred_bbox is None:
            continue

        weighted_iou_sum += weight * bbox_iou(gt_bbox, pred_bbox)

    weighted_avg_iou = weighted_iou_sum / (weight_sum)

    extra_pred_regions = set(pred_bbox_dict) - set(gt_bbox_dict)
    fp_penalty = len(extra_pred_regions)

    reward = weighted_avg_iou - fp_penalty_lambda * fp_penalty

    return float(reward)


# def multi_label_f1_score(pred_labels: List[int], target_labels: List[int]) -> float:
#     """
#     Compute binary multi-label F1 from two aligned 0/1 label vectors.
#     """
#     if not pred_labels or not target_labels or len(pred_labels) != len(target_labels):
#         raise ValueError("F1 score requires non-empty aligned label vectors.")

#     tp = sum(int(pred == 1 and target == 1) for pred, target in zip(pred_labels, target_labels))
#     fp = sum(int(pred == 1 and target == 0) for pred, target in zip(pred_labels, target_labels))
#     fn = sum(int(pred == 0 and target == 1) for pred, target in zip(pred_labels, target_labels))

#     if tp == 0:
#         return 1.0 if (fp == 0 and fn == 0) else 0.0

#     precision = float(tp) / float(tp + fp)
#     recall = float(tp) / float(tp + fn)
#     return float(2.0 * precision * recall / (precision + recall))


def multi_label_f1_score(pred_labels: List[int],target_labels: List[int]) -> float:
    """
    Compute class-frequency-weighted binary multi-label F1 from aligned 0/1 labels.

    Weights are normalized to have mean 1:
        p_c = n_c^+ / sum_j n_j^+
        w_c = (1 / p_c^alpha) / mean_j(1 / p_j^alpha)
    """
    alpha = 0.5

    #positive_counts = [29449, 43528, 42613, 6577, 21952, 9006, 14808, 44209, 5508, 37209, 3950, 6374, 45398, 36727]
    positive_counts = [2624, 7077, 10854, 2130, 2488, 1360, 3141, 8480, 302, 4418, 1300, 1479, 4757, 4664]

    if not pred_labels or not target_labels or len(pred_labels) != len(target_labels):
        raise ValueError("Weighted F1 score requires non-empty aligned label vectors.")

    
    positive_counts = positive_counts
    
    if len(positive_counts) != len(pred_labels):
        raise ValueError("Weighted F1 score requires positive counts aligned with label vectors.")

    total_positive = float(sum(positive_counts))

    inverse_freq = [
        1.0 / ((float(count) / total_positive) ** float(alpha))
        for count in positive_counts
    ]
    mean_inverse_freq = sum(inverse_freq) / float(len(inverse_freq))
    weights = [value / mean_inverse_freq for value in inverse_freq]

    tp_w = sum(
        weight
        for pred, target, weight in zip(pred_labels, target_labels, weights)
        if pred == 1 and target == 1
    )
    fp_w = sum(
        weight
        for pred, target, weight in zip(pred_labels, target_labels, weights)
        if pred == 1 and target == 0
    )
    fn_w = sum(
        weight
        for pred, target, weight in zip(pred_labels, target_labels, weights)
        if pred == 0 and target == 1
    )

    denominator = (2.0 * tp_w) + fp_w + fn_w
    if denominator == 0.0:
        return 1.0
    return float((2.0 * tp_w) / denominator)

def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _replace_box_image_block(match: re.Match[str]) -> str:
    before = match.string[: match.start()]
    after = match.string[match.end() :]
    if before and after and before[-1] not in "\r\n" and after[0] not in "\r\n":
        return " "
    return "\n\n"


def _clean_think_text(raw_think: str) -> str:
    cleaned = _BOX_IMAGE_BLOCK_PATTERN.sub(_replace_box_image_block, raw_think)
    cleaned = _IMAGE_TAG_WITH_SURROUNDING_WHITESPACE_PATTERN.sub(" ", cleaned)
    cleaned = _RESIDUAL_OBJECT_REF_LINE_PATTERN.sub("", cleaned)
    cleaned = _RESIDUAL_BOX_LINE_PATTERN.sub("", cleaned)
    cleaned = _RESIDUAL_IMAGE_LINE_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def parse_reward_text(text: str) -> Dict[str, Any]:
    """Split model text into bbox dict, cleaned think text, and final answer."""
    content = text or ""
    bbox_dict = {
        match.group(1).strip(): [int(coord) for coord in match.groups()[1:]]
        for match in _OBJECT_REF_BOX_PATTERN.finditer(content)
    }

    think_text = ""
    answer = content.strip()
    think_match = _THINK_PATTERN.search(content)
    if think_match:
        raw_think = think_match.group(1)
        think_text = _clean_think_text(raw_think)
        answer = content[think_match.end() :].strip()

    return {
        "bbox_dict": bbox_dict,
        "think_text": think_text,
        "answer": answer,
    }


def _fallback_reward(solution_str: str, ground_truth: str) -> list[float]:
    solution = solution_str or ""
    target = ground_truth or ""
    if target.strip():
        score = 1.0 if _normalize_text(solution) == _normalize_text(target) else 0.0
    else:
        score = 1.0 if solution.strip() else -1.0
    return [float(score), float(score), float(score), float(score)]


def _get_answer_chexbert_scorer():
    global _ANSWER_CHEXBERT_SCORER, _ANSWER_CHEXBERT_IMPORT_ERROR

    if _ANSWER_CHEXBERT_SCORER is not None:
        return _ANSWER_CHEXBERT_SCORER
    if _ANSWER_CHEXBERT_IMPORT_ERROR is not None:
        raise _ANSWER_CHEXBERT_IMPORT_ERROR

    try:
        print(
            f"[reward] initializing answer CheXbert on {_ANSWER_CHEXBERT_DEVICE}",
            flush=True,
        )
        _reward_log("answer_chexbert: init start")
        with _temporarily_disable_deepspeed_zero_init("answer_chexbert"):
            _ANSWER_CHEXBERT_SCORER = CheXbertLabeler(device=_ANSWER_CHEXBERT_DEVICE)
        _reward_log("answer_chexbert: scorer object constructed")
        print("[reward] answer CheXbert ready", flush=True)
    except Exception as exc:
        _ANSWER_CHEXBERT_IMPORT_ERROR = exc
        warnings.warn(
            f"Falling back to exact-match reward because answer CheXbert initialization failed: {exc}",
            RuntimeWarning,
        )
        raise
    return _ANSWER_CHEXBERT_SCORER


def _get_reasoning_chexbert_scorer():
    global _REASONING_CHEXBERT_SCORER, _REASONING_CHEXBERT_IMPORT_ERROR

    if _REASONING_CHEXBERT_SCORER is not None:
        return _REASONING_CHEXBERT_SCORER
    if _REASONING_CHEXBERT_IMPORT_ERROR is not None:
        raise _REASONING_CHEXBERT_IMPORT_ERROR

    try:
        print(
            f"[reward] initializing reasoning CheXbert on {_REASONING_CHEXBERT_DEVICE}",
            flush=True,
        )
        _reward_log("reasoning_chexbert: init start")
        with _temporarily_disable_deepspeed_zero_init("reasoning_chexbert"):
            _REASONING_CHEXBERT_SCORER = ReasoningCheXbertLabeler(device=_REASONING_CHEXBERT_DEVICE)
        _reward_log("reasoning_chexbert: scorer object constructed")
        print("[reward] reasoning CheXbert ready", flush=True)
    except Exception as exc:
        _REASONING_CHEXBERT_IMPORT_ERROR = exc
        warnings.warn(
            f"Falling back to exact-match reward because reasoning CheXbert initialization failed: {exc}",
            RuntimeWarning,
        )
        raise
    return _REASONING_CHEXBERT_SCORER


def _get_answer_chexbert_labels(text: str) -> list[int]:
    normalized = _normalize_text(text)
    if normalized in _ANSWER_CHEXBERT_LABEL_CACHE:
        return _ANSWER_CHEXBERT_LABEL_CACHE[normalized]

    scorer = _get_answer_chexbert_scorer()
    labels = scorer.get_label(text, mode="rrg")
    _ANSWER_CHEXBERT_LABEL_CACHE[normalized] = labels
    return labels


def _get_answer_chexbert_labels_batch(texts: List[str]) -> List[list[int]]:
    if not texts:
        return []

    outputs: List[Optional[list[int]]] = [None for _ in texts]
    uncached_positions: Dict[str, List[int]] = {}
    uncached_texts: Dict[str, str] = {}
    for idx, text in enumerate(texts):
        normalized = _normalize_text(text)
        cached = _ANSWER_CHEXBERT_LABEL_CACHE.get(normalized)
        if cached is not None:
            outputs[idx] = cached
            continue
        uncached_positions.setdefault(normalized, []).append(idx)
        uncached_texts.setdefault(normalized, text)

    if uncached_texts:
        scorer = _get_answer_chexbert_scorer()
        normalized_keys = list(uncached_texts.keys())
        computed = scorer.get_labels(
            [uncached_texts[key] for key in normalized_keys],
            mode="rrg",
            batch_size=_ANSWER_CHEXBERT_BATCH_SIZE,
        )
        for normalized, labels in zip(normalized_keys, computed):
            _ANSWER_CHEXBERT_LABEL_CACHE[normalized] = labels
            for pos in uncached_positions.get(normalized, []):
                outputs[pos] = labels

    return [labels if labels is not None else [] for labels in outputs]


def _get_reasoning_chexbert_labels(text: str) -> list[int]:
    normalized = _normalize_text(text)
    if normalized in _REASONING_CHEXBERT_LABEL_CACHE:
        return _REASONING_CHEXBERT_LABEL_CACHE[normalized]

    scorer = _get_reasoning_chexbert_scorer()
    labels = scorer.get_label(text, mode="rrg")
    _REASONING_CHEXBERT_LABEL_CACHE[normalized] = labels
    return labels


def _get_reasoning_chexbert_labels_batch(texts: List[str]) -> List[list[int]]:
    if not texts:
        return []

    outputs: List[Optional[list[int]]] = [None for _ in texts]
    uncached_positions: Dict[str, List[int]] = {}
    uncached_texts: Dict[str, str] = {}
    for idx, text in enumerate(texts):
        normalized = _normalize_text(text)
        cached = _REASONING_CHEXBERT_LABEL_CACHE.get(normalized)
        if cached is not None:
            outputs[idx] = cached
            continue
        uncached_positions.setdefault(normalized, []).append(idx)
        uncached_texts.setdefault(normalized, text)

    if uncached_texts:
        scorer = _get_reasoning_chexbert_scorer()
        normalized_keys = list(uncached_texts.keys())
        computed = scorer.get_labels(
            [uncached_texts[key] for key in normalized_keys],
            mode="rrg",
            batch_size=_REASONING_CHEXBERT_BATCH_SIZE,
        )
        for normalized, labels in zip(normalized_keys, computed):
            _REASONING_CHEXBERT_LABEL_CACHE[normalized] = labels
            for pos in uncached_positions.get(normalized, []):
                outputs[pos] = labels

    return [labels if labels is not None else [] for labels in outputs]


def _get_radgraph_scorer():
    global _RADGRAPH_SCORER, _RADGRAPH_IMPORT_ERROR

    if _RADGRAPH_SCORER is not None:
        return _RADGRAPH_SCORER
    if _RADGRAPH_IMPORT_ERROR is not None:
        raise _RADGRAPH_IMPORT_ERROR

    try:
        from radgraph import F1RadGraph

        print(
            f"[reward] initializing RadGraph on {_RADGRAPH_DEVICE} "
            f"(cache_dir={_RADGRAPH_MODEL_CACHE_DIR})",
            flush=True,
        )
        _reward_log("radgraph: init start")
        with _temporarily_disable_deepspeed_zero_init("radgraph"):
            _patch_radgraph_cached_transformers_get()
            _RADGRAPH_SCORER = F1RadGraph(
                reward_level="all",
                model_type="radgraph-xl",
                model_cache_dir=_RADGRAPH_MODEL_CACHE_DIR,
                cuda=_radgraph_cuda_arg(_RADGRAPH_DEVICE),
            )
            _RADGRAPH_SCORER.eval()
            for param in _RADGRAPH_SCORER.parameters():
                param.requires_grad = False
            _empty_radgraph_cuda_cache()
        _reward_log("radgraph: scorer object constructed")
        print("[reward] RadGraph ready", flush=True)
    except Exception as exc:
        _RADGRAPH_IMPORT_ERROR = exc
        print(
            f"[reward] RadGraph initialization runtime context: {_current_dist_runtime_context()}",
            flush=True,
        )
        print("[reward] RadGraph initialization traceback begin", flush=True)
        print(traceback.format_exc().rstrip(), flush=True)
        print("[reward] RadGraph initialization traceback end", flush=True)
        warnings.warn(
            f"RadGraph reward disabled because F1RadGraph initialization failed: {exc}",
            RuntimeWarning,
        )
        raise
    return _RADGRAPH_SCORER


def _compute_answer_radgraph_reward(pred_answer: str, target_answer: str) -> float:
    cache_key = (_normalize_text(pred_answer), _normalize_text(target_answer))
    if cache_key in _RADGRAPH_SCORE_CACHE:
        return _RADGRAPH_SCORE_CACHE[cache_key]

    scorer = _get_radgraph_scorer()
    radgraph_scores = _run_radgraph_scorer(scorer, refs=[target_answer], hyps=[pred_answer])
    f1_scores = radgraph_scores[0]

    score = _mean_radgraph_f1(f1_scores)
    _RADGRAPH_SCORE_CACHE[cache_key] = score
    return score


def _compute_answer_radgraph_reward_batch(
    pred_answers: List[str],
    target_answers: List[str],
) -> List[float]:
    if len(pred_answers) != len(target_answers):
        raise ValueError("pred_answers and target_answers must have the same length.")
    if not pred_answers:
        return []

    outputs: List[Optional[float]] = [None for _ in pred_answers]
    uncached_positions: Dict[Tuple[str, str], List[int]] = {}
    uncached_pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for idx, (pred_answer, target_answer) in enumerate(zip(pred_answers, target_answers)):
        cache_key = (_normalize_text(pred_answer), _normalize_text(target_answer))
        cached = _RADGRAPH_SCORE_CACHE.get(cache_key)
        if cached is not None:
            outputs[idx] = cached
            continue
        uncached_positions.setdefault(cache_key, []).append(idx)
        uncached_pairs.setdefault(cache_key, (pred_answer, target_answer))

    if uncached_pairs:
        scorer = _get_radgraph_scorer()
        uncached_keys = list(uncached_pairs.keys())
        for start in range(0, len(uncached_keys), max(1, _RADGRAPH_BATCH_SIZE)):
            chunk_keys = uncached_keys[start : start + max(1, _RADGRAPH_BATCH_SIZE)]
            chunk_pairs = [uncached_pairs[key] for key in chunk_keys]
            refs = [pair[1] for pair in chunk_pairs]
            hyps = [pair[0] for pair in chunk_pairs]
            try:
                radgraph_scores = _run_radgraph_scorer(scorer, refs=refs, hyps=hyps)
                chunk_scores = _parse_radgraph_batch_output(radgraph_scores, len(chunk_pairs))
            except Exception:
                chunk_scores = [
                    _compute_answer_radgraph_reward(pred_answer=pred, target_answer=target)
                    for pred, target in chunk_pairs
                ]

            for cache_key, score in zip(chunk_keys, chunk_scores):
                _RADGRAPH_SCORE_CACHE[cache_key] = score
                for pos in uncached_positions.get(cache_key, []):
                    outputs[pos] = score

    return [float(score) if score is not None else 0.0 for score in outputs]


def _compute_answer_chexbert_components(
    pred_answer: str,
    target_answer: str,
) -> Tuple[List[int], List[int], float]:
    pred_labels = _get_answer_chexbert_labels(pred_answer)
    target_labels = _get_answer_chexbert_labels(target_answer)
    if not pred_labels or not target_labels or len(pred_labels) != len(target_labels):
        raise ValueError("CheXbert label extraction returned mismatched label vectors.")
    answer_reward = multi_label_f1_score(pred_labels, target_labels)
    return pred_labels, target_labels, answer_reward


def _compute_answer_chexbert_components_batch(
    pred_answers: List[str],
    target_answers: List[str],
) -> Tuple[List[List[int]], List[List[int]], List[float]]:
    pred_labels_batch = _get_answer_chexbert_labels_batch(pred_answers)
    target_labels_batch = _get_answer_chexbert_labels_batch(target_answers)

    answer_rewards: List[float] = []
    for pred_labels, target_labels in zip(pred_labels_batch, target_labels_batch):
        if not pred_labels or not target_labels or len(pred_labels) != len(target_labels):
            raise ValueError("CheXbert label extraction returned mismatched label vectors.")
        answer_rewards.append(multi_label_f1_score(pred_labels, target_labels))
    return pred_labels_batch, target_labels_batch, answer_rewards


def _prewarm_reward_backends():
    global _REWARD_BACKENDS_PREWARMED
    if _REWARD_BACKENDS_PREWARMED:
        return

    print("[reward] prewarming reward backends (serial)", flush=True)
    if _ENABLE_REASONING_CHEXBERT:
        _get_reasoning_chexbert_scorer()
    else:
        print("[reward] reasoning CheXbert disabled by QWENVL_ENABLE_REASONING_CHEXBERT", flush=True)
    _get_answer_chexbert_scorer()
    if _ENABLE_RADGRAPH:
        try:
            _get_radgraph_scorer()
        except Exception:
            print("[reward] RadGraph disabled after initialization failure", flush=True)
    else:
        print("[reward] RadGraph disabled by QWENVL_ENABLE_RADGRAPH", flush=True)
    _REWARD_BACKENDS_PREWARMED = True
    print("[reward] reward backend prewarm complete", flush=True)


def _compute_reward(solution_str: str, ground_truth: str) -> list[float]:
    parsed_solution = parse_reward_text(solution_str or "")
    parsed_target = parse_reward_text(ground_truth or "")

    if parsed_solution["think_text"] == "":
        return [float(0.0), float(0.0), float(0.0), float(0.0)]

    _prewarm_reward_backends()

    bbox_reward = bbox_reward_with_fp_penalty(
        parsed_target["bbox_dict"],
        parsed_solution["bbox_dict"],
    )
    _, target_labels, answer_chexbert_reward = _compute_answer_chexbert_components(
        parsed_solution["answer"],
        parsed_target["answer"],
    )

    answer_radgraph_reward = _compute_answer_radgraph_reward(
        parsed_solution["answer"],
        parsed_target["answer"],
    )


    
    pred_thinking_labels = _get_reasoning_chexbert_labels(parsed_solution["think_text"])
    if not pred_thinking_labels or len(pred_thinking_labels) != len(target_labels):
        raise ValueError("Reasoning CheXbert label extraction returned mismatched label vectors.")
    reasoning_reward = multi_label_f1_score(pred_thinking_labels, target_labels)


    return [
        float(bbox_reward),
        float(reasoning_reward),
        float(answer_chexbert_reward),
        float(answer_radgraph_reward),
    ]


def _compute_reward_batch(items: List[Dict[str, Any]]) -> List[list[float]]:
    parsed_solutions = [parse_reward_text((item.get("text") or "")) for item in items]
    parsed_targets = [parse_reward_text((item.get("ground_truth") or "")) for item in items]

    rewards: List[Optional[list[float]]] = [None for _ in items]
    active_indices: List[int] = []
    for idx, item in enumerate(items):
        target = item.get("ground_truth") or ""
        if not target.strip():
            rewards[idx] = _fallback_reward(item.get("text") or "", target)
            continue
        if parsed_solutions[idx]["think_text"] == "":
            rewards[idx] = [float(0.0), float(0.0), float(0.0), float(0.0)]
            continue
        active_indices.append(idx)

    if not active_indices:
        return [reward if reward is not None else [0.0, 0.0, 0.0, 0.0] for reward in rewards]

    _prewarm_reward_backends()

    reward_batch_size = max(1, _REWARD_BATCH_SIZE)
    for chunk_start in range(0, len(active_indices), reward_batch_size):
        chunk_indices = active_indices[chunk_start : chunk_start + reward_batch_size]
        active_pred_answers = [parsed_solutions[idx]["answer"] for idx in chunk_indices]
        active_target_answers = [parsed_targets[idx]["answer"] for idx in chunk_indices]
        active_reasoning_texts = [parsed_solutions[idx]["think_text"] for idx in chunk_indices]

        _, target_labels_batch, answer_chexbert_rewards = _compute_answer_chexbert_components_batch(
            active_pred_answers,
            active_target_answers,
        )
        answer_radgraph_rewards = _compute_answer_radgraph_reward_batch(
            active_pred_answers,
            active_target_answers,
        )
        pred_reasoning_labels_batch = _get_reasoning_chexbert_labels_batch(active_reasoning_texts)

        for batch_idx, sample_idx in enumerate(chunk_indices):
            target_labels = target_labels_batch[batch_idx]
            pred_reasoning_labels = pred_reasoning_labels_batch[batch_idx]
            if not pred_reasoning_labels or len(pred_reasoning_labels) != len(target_labels):
                raise ValueError("Reasoning CheXbert label extraction returned mismatched label vectors.")
            reasoning_reward = multi_label_f1_score(pred_reasoning_labels, target_labels)
            
            bbox_reward = bbox_reward_with_fp_penalty(
                parsed_targets[sample_idx]["bbox_dict"],
                parsed_solutions[sample_idx]["bbox_dict"],
            )

            rewards[sample_idx] = [
                float(bbox_reward),
                float(reasoning_reward),
                float(answer_chexbert_rewards[batch_idx]),
                float(answer_radgraph_rewards[batch_idx]),
            ]

    return [reward if reward is not None else [0.0, 0.0, 0.0, 0.0] for reward in rewards]


def default_reward(solution_str: str, ground_truth: str, meta: Optional[Dict[str, Any]] = None) -> float:
    """Compute reward from CheXbert label agreement between prediction and GT.

    Falls back to the original conservative reward when GT is empty or CheXbert
    cannot be initialized in the current runtime.
    """
    target = ground_truth or ""
    if not target.strip():
        return _fallback_reward(solution_str, ground_truth)

    try:
        return _compute_reward(solution_str or "", target)
    except Exception:
        return _fallback_reward(solution_str, ground_truth)


def default_reward_batch(items: List[Dict[str, Any]]) -> List[list[float]]:
    if not items:
        return []
    try:
        return _compute_reward_batch(items)
    except Exception:
        return [
            default_reward(
                item.get("text") or "",
                item.get("ground_truth") or "",
                item.get("meta"),
            )
            for item in items
        ]


def load_reward_function(
    module_name: Optional[str],
    function_name: str = "compute_reward",
) -> Callable[[str, str, Optional[Dict[str, Any]]], float]:
    """Load user reward function. If not provided, fallback to default_reward."""
    if not module_name:
        return default_reward
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise TypeError(f"{module_name}.{function_name} is not callable")
    return fn


def load_reward_batch_function(
    module_name: Optional[str],
    function_name: str = "compute_reward_batch",
) -> Optional[Callable[[List[Dict[str, Any]]], List[float]]]:
    if not module_name:
        return default_reward_batch
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name, None)
    if fn is None:
        return None
    if not callable(fn):
        raise TypeError(f"{module_name}.{function_name} is not callable")
    return fn
