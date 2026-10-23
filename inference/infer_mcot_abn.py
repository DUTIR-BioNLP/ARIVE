"""Training-aligned ARIVE inference for the MIMIC-ABN test split.

This entry point reuses :mod:`inference.infer_mcot` for model loading and the
iterative external-vision rollout.  Its dataset adapter follows the original
MIMIC-ABN experiment: test membership comes from the abnormal annotation,
whereas prior findings are resolved from the full Prior-RRG annotation.

Examples:
    QWENVL_INFER_PREFLIGHT_ONLY=1 python inference/infer_mcot_abn.py
    CUDA_VISIBLE_DEVICES=0 python inference/infer_mcot_abn.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Set


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import (  # noqa: E402
    IMAGE_ROOT,
    INFERENCE_ABN_ANNOTATION,
    INFERENCE_ABN_REPORTS,
    INFERENCE_CLINICAL_CONTEXT,
    INFERENCE_PRIOR_ANNOTATION,
)


SYSTEM_PROMPT = (
    "You are an assistant in radiology, responsible for analyzing medical "
    "imaging studies and generating detailed, structured, and accurate "
    "radiology reports."
)
MCOT_INSTRUCTION_PROMPT = (
    "Before answering, think step by step internally. In your internal thinking, "
    "output the bounding box of the key diagnostic region in the format "
    "<|object_ref_start|>anatomical object name<|object_ref_end|>"
    "<|box_start|>x1 y1 x2 y2<|box_end|>. Generate a comprehensive and detailed "
    "description of abnormal findings only based on the current chest X-ray "
    "image and the indication."
)
MCOT_INSTRUCTION_PRIOR_PROMPT = (
    f"{MCOT_INSTRUCTION_PROMPT} Include a thorough comparison with a prior chest "
    "X-ray, emphasizing any significant changes, progression, or improvement."
)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off")


def _prepend_env_path(name: str, entry: Path) -> bool:
    entry_text = str(entry)
    old_entries = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    new_entries = [entry_text, *(part for part in old_entries if part != entry_text)]
    new_value = os.pathsep.join(new_entries)
    changed = os.environ.get(name, "") != new_value
    os.environ[name] = new_value
    return changed


def _ensure_qwenvl_runtime_before_native_imports() -> None:
    """Restart once with an explicitly configured Qwen-VL environment."""
    configured_env = os.environ.get("QWENVL_ENV_DIR", "").strip()
    if not configured_env:
        return
    env_dir = Path(configured_env).expanduser().resolve()
    env_python = env_dir / "bin/python"
    env_lib = env_dir / "lib"
    env_libstdcxx = env_lib / "libstdc++.so.6"
    if not env_python.is_file() or not os.access(env_python, os.X_OK):
        raise RuntimeError(f"qwenvl Python is not executable: {env_python}")

    os.environ["QWENVL_ENV_DIR"] = str(env_dir)
    _prepend_env_path("PATH", env_dir / "bin")
    library_path_changed = (
        _prepend_env_path("LD_LIBRARY_PATH", env_lib)
        if env_libstdcxx.is_file()
        else False
    )
    wrong_python = Path(sys.executable).resolve() != env_python.resolve()
    if library_path_changed or wrong_python:
        print(f"[Info] Restarting with qwenvl Python runtime from {env_dir}", flush=True)
        argv = [str(env_python), str(Path(__file__).resolve()), *sys.argv[1:]]
        os.execve(str(env_python), argv, os.environ.copy())


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _load_json_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def normalize_image_path(image_path: str) -> str:
    """Normalize annotation and report paths to ``files/...`` lookup keys."""
    if not isinstance(image_path, str) or not image_path.strip():
        raise TypeError("image paths must be non-empty strings")
    normalized = image_path.strip().replace("\\", "/")
    if normalized.startswith("files/"):
        return normalized
    marker = "/files/"
    if marker in normalized:
        return f"files/{normalized.split(marker, 1)[1]}"
    return f"files/{normalized.lstrip('/')}"


def _image_values(value: Any) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value]
    if isinstance(value, list) and value and all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return list(value)
    raise TypeError("image_path must be a non-empty string or list of strings")


def _sample_images(sample: Mapping[str, Any]) -> List[str]:
    value = sample.get("image")
    if isinstance(value, str) and value.strip():
        return [normalize_image_path(value)]
    if isinstance(value, list) and value and all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return [normalize_image_path(item) for item in value]
    raise TypeError("each evaluation record needs a non-empty image string or list")


def _add_image_findings(
    image_to_findings: MutableMapping[str, str],
    image_path: Any,
    findings: Any,
) -> None:
    if not image_path or not isinstance(findings, str) or not findings.strip():
        return
    for path in _image_values(image_path):
        image_to_findings[normalize_image_path(path)] = findings


def _add_prior_study_findings(
    image_to_findings: MutableMapping[str, str], prior_study: Any
) -> None:
    if isinstance(prior_study, dict):
        if "image_path" in prior_study and "findings" in prior_study:
            _add_image_findings(
                image_to_findings,
                prior_study.get("image_path"),
                prior_study.get("findings"),
            )
        for value in prior_study.values():
            _add_prior_study_findings(image_to_findings, value)
    elif isinstance(prior_study, list):
        for value in prior_study:
            _add_prior_study_findings(image_to_findings, value)


def build_image_to_findings(data_split: Mapping[str, Any]) -> Dict[str, str]:
    """Index anchor and recursively nested prior images across every split."""
    image_to_findings: Dict[str, str] = {}
    for split_name, split_data in data_split.items():
        if not isinstance(split_data, list):
            raise TypeError(f"annotation split {split_name!r} must be a JSON list")
        for index, instance in enumerate(split_data):
            if not isinstance(instance, dict):
                raise TypeError(f"{split_name}[{index}] must be a JSON object")
            anchor_scan = instance.get("anchor_scan")
            if not isinstance(anchor_scan, dict) or "image_path" not in anchor_scan:
                raise KeyError(f"{split_name}[{index}] is missing anchor_scan.image_path")
            _add_image_findings(
                image_to_findings,
                anchor_scan["image_path"],
                instance.get("findings", ""),
            )
            _add_prior_study_findings(
                image_to_findings,
                instance.get("prior_study"),
            )
    return image_to_findings


def build_abn_test_keys(data_split: Mapping[str, Any]) -> Set[str]:
    test_data = data_split.get("test")
    if not isinstance(test_data, list):
        raise TypeError("MIMIC-ABN annotation must contain a test JSON list")
    keys: Set[str] = set()
    for index, instance in enumerate(test_data):
        if not isinstance(instance, dict):
            raise TypeError(f"test[{index}] must be a JSON object")
        anchor_scan = instance.get("anchor_scan")
        if not isinstance(anchor_scan, dict) or "image_path" not in anchor_scan:
            raise KeyError(f"test[{index}] is missing anchor_scan.image_path")
        keys.add(normalize_image_path(_image_values(anchor_scan["image_path"])[0]))
    if not keys:
        raise ValueError("the MIMIC-ABN test split is empty")
    return keys


def select_abn_test_samples(
    reports: Sequence[Dict[str, Any]], test_keys: Set[str]
) -> List[Dict[str, Any]]:
    selected = [sample for sample in reports if _sample_images(sample)[0] in test_keys]
    if not selected:
        raise RuntimeError("No MIMIC-ABN test samples matched the evaluation report list")
    question_ids = [sample.get("question_id") for sample in selected]
    if any(question_id is None for question_id in question_ids):
        raise KeyError("every selected evaluation row must contain question_id")
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("selected evaluation rows contain duplicate question_id values")
    return selected


def _study_id(image_path: str) -> str:
    for part in Path(image_path).parts:
        if re.fullmatch(r"s\d+", part):
            return part
    raise ValueError(f"cannot infer a MIMIC study id from image path: {image_path}")


def _get_indication(image_path: str, indications: Mapping[str, Any]) -> str:
    context = indications.get(_study_id(image_path), {})
    if not isinstance(context, dict):
        return ""
    indication = context.get("Indication", "")
    return indication.strip() if isinstance(indication, str) else ""


def _absolute_image_path(image_root: Path, relative_path: str) -> str:
    path = Path(relative_path)
    return str(path if path.is_absolute() else image_root / path)


def _normalize_user_content_like_training(
    content: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in content:
        if item.get("type") != "text":
            normalized.append(item)
            continue
        text = item.get("text")
        if not isinstance(text, str):
            raise TypeError("user text content must be a string")
        if text.strip():
            normalized.append({**item, "text": text.strip()})
    return normalized


def data_construct(
    sample: Mapping[str, Any],
    image_to_findings: Mapping[str, str],
    indications: Mapping[str, Any],
    image_root: Path,
) -> Dict[str, Any]:
    """Build an abnormal-only prompt with training-equivalent image ordering."""
    images = _sample_images(sample)
    current_image = images[0]
    prior_image = images[1] if len(images) > 1 and images[1] != current_image else None
    prior_findings = image_to_findings.get(prior_image, "") if prior_image else ""
    indication = _get_indication(current_image, indications)

    opening = SYSTEM_PROMPT
    if indication:
        opening += f"\n\nIndication: {indication}"
    if prior_findings:
        content = [
            {"type": "text", "text": f"{opening}\nPrior chest X-ray:\n"},
            {
                "type": "image",
                "image": _absolute_image_path(image_root, prior_image),
            },
            {
                "type": "text",
                "text": f"\nPrior abnormal findings:{prior_findings}\nCurrent chest X-ray:\n",
            },
            {
                "type": "image",
                "image": _absolute_image_path(image_root, current_image),
            },
            {"type": "text", "text": f"\n{MCOT_INSTRUCTION_PRIOR_PROMPT}"},
        ]
    else:
        content = [
            {"type": "text", "text": f"{opening}\n\nCurrent chest X-ray:\n"},
            {
                "type": "image",
                "image": _absolute_image_path(image_root, current_image),
            },
            {"type": "text", "text": f"\n{MCOT_INSTRUCTION_PROMPT}"},
        ]
    return {
        "role": "user",
        "content": _normalize_user_content_like_training(content),
    }


def _validate_paths(
    paths: Mapping[str, Path], image_root: Path, *, require_image_root: bool
) -> None:
    for variable, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{variable} must point to an existing file: {path}")
    if require_image_root and not image_root.is_dir():
        raise FileNotFoundError(
            "QWENVL_INFER_MIMIC_CXR_ROOT must point to the MIMIC-CXR image directory: "
            f"{image_root}"
        )


def _validate_message_images(messages: Sequence[Sequence[Mapping[str, Any]]]) -> None:
    missing: List[str] = []
    image_count = 0
    for message_group in messages:
        for message in message_group:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_count += 1
                    image_path = item.get("image")
                    if not isinstance(image_path, str) or not Path(image_path).is_file():
                        missing.append(str(image_path))
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)}/{image_count} selected image references are missing; "
            f"first paths: {preview}"
        )
    print(f"[Info] Verified {image_count} selected image references", flush=True)


def main() -> None:
    start_time = time.time()
    report_path = Path(
        os.environ.get("QWENVL_INFER_REPORT_PATH", str(INFERENCE_ABN_REPORTS))
    ).expanduser()
    abn_annotation_path = Path(
        os.environ.get(
            "QWENVL_INFER_ABN_ANNOTATION_PATH", str(INFERENCE_ABN_ANNOTATION)
        )
    ).expanduser()
    prior_annotation_path = Path(
        os.environ.get(
            "QWENVL_INFER_PRIOR_ANNOTATION_PATH", str(INFERENCE_PRIOR_ANNOTATION)
        )
    ).expanduser()
    context_path = Path(
        os.environ.get(
            "QWENVL_INFER_CLINICAL_CONTEXT_PATH", str(INFERENCE_CLINICAL_CONTEXT)
        )
    ).expanduser()
    image_root = Path(
        os.environ.get("QWENVL_INFER_MIMIC_CXR_ROOT", str(IMAGE_ROOT))
    ).expanduser()
    preflight_only = _bool_env("QWENVL_INFER_PREFLIGHT_ONLY")
    check_images = _bool_env("QWENVL_INFER_PREFLIGHT_CHECK_IMAGES", default=True)
    if not preflight_only:
        # Restart before loading the two large annotation files so a configured
        # environment switch does not parse and allocate them twice.
        _ensure_qwenvl_runtime_before_native_imports()

    _validate_paths(
        {
            "QWENVL_INFER_REPORT_PATH": report_path,
            "QWENVL_INFER_ABN_ANNOTATION_PATH": abn_annotation_path,
            "QWENVL_INFER_PRIOR_ANNOTATION_PATH": prior_annotation_path,
            "QWENVL_INFER_CLINICAL_CONTEXT_PATH": context_path,
        },
        image_root,
        require_image_root=not preflight_only or check_images,
    )

    report_list = _load_jsonl(report_path)
    if not report_list:
        raise ValueError("the evaluation report JSONL is empty")
    abn_data_split = _load_json_object(abn_annotation_path)
    prior_data_split = _load_json_object(prior_annotation_path)
    indications = _load_json_object(context_path)

    test_keys = build_abn_test_keys(abn_data_split)
    del abn_data_split
    image_to_findings = build_image_to_findings(prior_data_split)
    del prior_data_split
    test_list = select_abn_test_samples(report_list, test_keys)
    messages = [
        [data_construct(sample, image_to_findings, indications, image_root)]
        for sample in test_list
    ]
    missing_prior_count = sum(
        1
        for sample in test_list
        if len(_sample_images(sample)) > 1
        and _sample_images(sample)[1] != _sample_images(sample)[0]
        and _sample_images(sample)[1] not in image_to_findings
    )
    print(
        "[Info] MIMIC-ABN selection: "
        f"evaluation_rows={len(report_list)}, abn_test_keys={len(test_keys)}, "
        f"matched_rows={len(test_list)}, prior_index_entries={len(image_to_findings)}",
        flush=True,
    )
    print(
        f"[Info] Missing prior converted to current-only: {missing_prior_count}",
        flush=True,
    )
    # Messages contain the resolved text and paths, so the large lookup tables
    # and unmatched evaluation rows can be released before model allocation.
    del image_to_findings, indications, test_keys, report_list
    if check_images:
        _validate_message_images(messages)
    if preflight_only:
        print(
            f"[Info] MIMIC-ABN preflight passed in {time.time() - start_time:.2f}s; "
            "no model was loaded and no output was written",
            flush=True,
        )
        return

    # Import native Torch/vLLM dependencies only after the optional runtime
    # restart, while keeping dataset helpers importable in lightweight tests.
    from inference import infer_mcot as core

    base_model = os.environ.get(
        "QWENVL_INFER_BASE_MODEL", "Qwen/Qwen3-VL-4B-Instruct"
    )
    vision_checkpoint = os.environ.get("QWENVL_INFER_VISION_CHECKPOINT", "")
    vllm_checkpoint = os.environ.get("QWENVL_INFER_VLLM_CHECKPOINT", "")
    for variable, raw_path in {
        "QWENVL_INFER_VISION_CHECKPOINT": vision_checkpoint,
        "QWENVL_INFER_VLLM_CHECKPOINT": vllm_checkpoint,
    }.items():
        if not raw_path or not Path(raw_path).is_dir():
            raise FileNotFoundError(f"{variable} must point to a checkpoint directory")

    max_model_len = int(os.environ.get("QWENVL_INFER_MAX_MODEL_LEN", "4096"))
    max_total_action_tokens = int(
        os.environ.get("QWENVL_INFER_MAX_TOTAL_ACTION_TOKENS", "6144")
    )
    max_pause_times = int(os.environ.get("QWENVL_INFER_MAX_PAUSE_TIMES", "64"))
    pause_image_encode_batch_size = int(
        os.environ.get("QWENVL_INFER_PAUSE_IMAGE_ENCODE_BATCH_SIZE", "32")
    )
    if max_model_len <= 0 or max_total_action_tokens <= 0:
        raise ValueError("model length and total action-token limit must be > 0")
    if max_pause_times < 0 or pause_image_encode_batch_size <= 0:
        raise ValueError("pause limit must be >= 0 and image encode batch size must be > 0")

    temperature = core._parse_temperature(
        os.environ.get("QWENVL_INFER_TEMPERATURES", "0.7")
    )
    top_p = float(os.environ.get("QWENVL_INFER_TOP_P", "0.8"))
    top_k = int(os.environ.get("QWENVL_INFER_TOP_K", "-1"))
    max_tokens_per_round = int(
        os.environ.get("QWENVL_INFER_MAX_TOKENS_PER_ROUND", "3072")
    )
    presence_penalty = core._parse_presence_penalty(
        os.environ.get(
            "QWENVL_INFER_PRESENCE_PENALTIES",
            os.environ.get("QWENVL_INFER_PRESENCE_PENALTY", "1.5"),
        )
    )
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("QWENVL_INFER_TOP_P must be in (0, 1]")
    if max_tokens_per_round <= 0:
        raise ValueError("QWENVL_INFER_MAX_TOKENS_PER_ROUND must be > 0")

    core.check_vllm_mm_embed_capability()
    print(f"[Info] Loading external vision model from {vision_checkpoint}", flush=True)
    vision_model = core.load_external_vision_model(base_model, vision_checkpoint)
    processor = core.AutoProcessor.from_pretrained(
        vllm_checkpoint,
        fix_mistral_regex=True,
    )
    core.patch_vllm_qwen3vl_mm_embed_field_sizes()
    inputs = core._prepare_exact_inputs(messages, processor, vision_model)

    tensor_parallel_size = int(
        os.environ.get("QWENVL_INFER_TP_SIZE", str(core.torch.cuda.device_count()))
    )
    if tensor_parallel_size <= 0:
        raise ValueError("QWENVL_INFER_TP_SIZE must be > 0 and CUDA must be available")
    engine_seed = int(os.environ.get("QWENVL_INFER_SEED", "3407"))
    llm = core.LLM(
        model=vllm_checkpoint,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=float(
            os.environ.get("QWENVL_INFER_GPU_MEMORY_UTILIZATION", "0.24")
        ),
        max_num_seqs=int(os.environ.get("QWENVL_INFER_MAX_NUM_SEQS", "16")),
        max_model_len=max_model_len,
        enable_mm_embeds=True,
        limit_mm_per_prompt={
            "image": int(os.environ.get("QWENVL_INFER_LIMIT_MM_IMAGE", "256"))
        },
        enable_expert_parallel=False,
        seed=engine_seed,
    )
    stop_token_ids = core._true_stop_token_ids(processor, vision_model)
    pause_token_id = int(os.environ.get("QWENVL_INFER_PAUSE_TOKEN_ID", "151649"))

    sampling_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens_per_round,
        "repetition_penalty": 1.0,
        "presence_penalty": presence_penalty,
    }
    request_seed = os.environ.get("QWENVL_INFER_REQUEST_SEED", "").strip()
    if request_seed:
        sampling_kwargs["seed"] = int(request_seed)
    sampling_params = core.SamplingParams(**sampling_kwargs)

    output_dir = Path(
        os.environ.get(
            "QWENVL_INFER_OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "inference_abn")
        )
    )
    output_prefix = os.environ.get("QWENVL_INFER_OUTPUT_PREFIX", "abn")
    output_path = output_dir / (
        f"{output_prefix}_{core._temperature_tag(temperature)}_{engine_seed}_"
        f"{core._presence_penalty_tag(presence_penalty)}.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not _bool_env("QWENVL_INFER_OVERWRITE"):
        raise FileExistsError(
            f"refusing to overwrite {output_path}; set QWENVL_INFER_OVERWRITE=1"
        )

    print(
        "[Info] Starting MIMIC-ABN generation: "
        f"samples={len(inputs)}, temperature={temperature}, top_p={top_p}, "
        f"top_k={top_k}, presence_penalty={presence_penalty}, "
        f"engine_seed={engine_seed}, request_seed={request_seed or 'unset'}",
        flush=True,
    )
    exact_results = core.iterative_generate_dynamic_batch_external(
        llm=llm,
        processor=processor,
        vision_model=vision_model,
        inputs=inputs,
        messages=messages,
        report_list=test_list,
        base_params=sampling_params,
        pause_token_id=pause_token_id,
        true_stop_token_id=stop_token_ids,
        max_pause_times=max_pause_times,
        max_total_new_tokens=max_total_action_tokens,
        max_model_len=max_model_len,
        pause_image_encode_batch_size=pause_image_encode_batch_size,
    )
    if len(exact_results) != len(inputs):
        raise RuntimeError(
            "Full-dataset rollout returned a mismatched result count: "
            f"expected={len(inputs)}, got={len(exact_results)}"
        )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(core._public_results(exact_results), handle, ensure_ascii=False, indent=2)
    print(
        f"[Info] Wrote {len(exact_results)} results to {output_path}; "
        f"elapsed={time.time() - start_time:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
