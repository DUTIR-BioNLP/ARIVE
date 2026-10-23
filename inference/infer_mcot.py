"""Training-aligned external-vision ARIVE inference.

The script is self-contained apart from the rollout implementation shared with
GRPO training. Initial images and later crop observations use the same training
preprocessing path, sampled actions are carried between rounds as token IDs,
and injected observations are explicitly masked and included in context-length
accounting.

Example:
    CUDA_VISIBLE_DEVICES=0 python inference/infer_mcot.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _prepend_env_path(name: str, entry: Path) -> bool:
    """Prepend one path entry and report whether the variable changed."""
    entry_text = str(entry)
    old_entries = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    new_entries = [entry_text, *(part for part in old_entries if part != entry_text)]
    new_value = os.pathsep.join(new_entries)
    changed = os.environ.get(name, "") != new_value
    os.environ[name] = new_value
    return changed


def _ensure_qwenvl_runtime_before_native_imports() -> None:
    """Restart once with the configured Python and optional C++ runtime."""
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
    ld_library_path_changed = (
        _prepend_env_path("LD_LIBRARY_PATH", env_lib)
        if env_libstdcxx.is_file()
        else False
    )
    wrong_python = Path(sys.executable).resolve() != env_python.resolve()

    # glibc determines native-library search paths when a process starts.  An
    # in-process LD_LIBRARY_PATH assignment is therefore insufficient for
    # extensions already loaded by Torch; replace this process before importing
    # Torch/vLLM so flash-attn sees qwenvl's CXXABI_1.3.15 runtime.
    if ld_library_path_changed or wrong_python:
        print(
            f"[Info] Restarting with qwenvl Python runtime from {env_dir}",
            flush=True,
        )
        argv = [str(env_python), str(Path(__file__).resolve()), *sys.argv[1:]]
        os.execve(str(env_python), argv, os.environ.copy())


if __name__ == "__main__":
    _ensure_qwenvl_runtime_before_native_imports()

# vLLM reads this during import, including imports through user_rollout.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)
from vllm import LLM, SamplingParams  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import (  # noqa: E402
    IMAGE_ROOT,
    INFERENCE_CLINICAL_CONTEXT,
    INFERENCE_REFERENCE_REPORTS,
    INFERENCE_REPORTS,
)

# This is the same rollout implementation used by train_grpo_verl.py.  It
# also owns image preprocessing, visual-branch dispatch, the external-embed
# compatibility patch, exact action token tracking, and context accounting.
from qwenvl.rl_grpo_verl.user_rollout import (  # noqa: E402
    attach_local_vision_branch,
    check_vllm_mm_embed_capability,
    iterative_generate_dynamic_batch_external,
    patch_vllm_qwen3vl_mm_embed_field_sizes,
    prepare_inputs_for_vllm_external,
    select_anchor_image_path,
)
from qwenvl.train.checkpoint_loader import (  # noqa: E402
    load_vision_checkpoint_into_model,
)


REPORT_PATH = os.environ.get(
    "QWENVL_INFER_REPORT_PATH",
    str(INFERENCE_REPORTS),
)
REFERENCE_REPORT_PATH = os.environ.get(
    "QWENVL_INFER_REFERENCE_REPORT_PATH",
    str(INFERENCE_REFERENCE_REPORTS),
)
CLINICAL_CONTEXT_PATH = os.environ.get(
    "QWENVL_INFER_CLINICAL_CONTEXT_PATH",
    str(INFERENCE_CLINICAL_CONTEXT),
)
OUTPUT_DIR = os.environ.get(
    "QWENVL_INFER_OUTPUT_DIR",
    str(PROJECT_ROOT / "outputs" / "inference"),
)
MIMIC_CXR_ROOT = os.environ.get(
    "QWENVL_INFER_MIMIC_CXR_ROOT",
    str(IMAGE_ROOT),
)

MAX_MODEL_LEN = int(os.environ.get("QWENVL_INFER_MAX_MODEL_LEN", "4096"))
MAX_TOTAL_ACTION_TOKENS = int(
    os.environ.get("QWENVL_INFER_MAX_TOTAL_ACTION_TOKENS", "6144")
)
MAX_PAUSE_TIMES = int(os.environ.get("QWENVL_INFER_MAX_PAUSE_TIMES", "64"))
PREPARE_LOG_INTERVAL = int(
    os.environ.get("QWENVL_INFER_PREPARE_LOG_INTERVAL", "50")
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
    "description of findings based on the current chest X-ray image and the "
    "indication."
)
MCOT_INSTRUCTION_PRIOR_PROMPT = (
    f"{MCOT_INSTRUCTION_PROMPT} Include a thorough comparison with a prior chest "
    "X-ray, emphasizing any significant changes, progression, or improvement."
)


def _parse_temperature(raw: str) -> float:
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(
            "QWENVL_INFER_TEMPERATURES must contain exactly one number."
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("The inference temperature must be finite and > 0.")
    return value


def _parse_presence_penalty(raw: str) -> float:
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(
            "QWENVL_INFER_PRESENCE_PENALTIES must contain exactly one number."
        ) from exc
    if not math.isfinite(value):
        raise ValueError("The inference presence penalty must be finite.")
    if not -2.0 <= value <= 2.0:
        raise ValueError("The inference presence penalty must be in [-2, 2].")
    return value


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{path} must contain one JSON object per non-empty line")
    return rows


def _sample_images(sample: Dict[str, Any]) -> List[str]:
    value = sample.get("image")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return list(value)
    raise TypeError("each inference record needs a non-empty image string or image list")


def _sample_report(sample: Dict[str, Any]) -> str:
    for key in ("findings", "report"):
        value = sample.get(key)
        if isinstance(value, str):
            return value
    raise KeyError("each inference record needs a findings or report string")


def _study_id(image_path: str) -> str:
    for part in Path(image_path).parts:
        if re.fullmatch(r"s\d+", part):
            return part
    raise ValueError(f"cannot infer a MIMIC study id from image path: {image_path}")


def _is_missing_scalar(value: Any) -> bool:
    """Match the missing-value cases produced by the JSON context file."""
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _mimic_image_path(relative_path: str) -> str:
    path = Path(relative_path)
    return str(path if path.is_absolute() else Path(MIMIC_CXR_ROOT) / path)


def _normalize_user_content_like_training(
    content: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply data_processor._build_messages text-segment semantics."""
    normalized: List[Dict[str, Any]] = []
    for item in content:
        if item.get("type") != "text":
            normalized.append(item)
            continue

        text = item.get("text")
        if not isinstance(text, str):
            raise TypeError(f"User text content must be a string, got {type(text).__name__}.")
        stripped_text = text.strip()
        if stripped_text:
            normalized.append({**item, "text": stripped_text})
    return normalized


def data_construct(
    sample: Dict[str, Any],
    indications: Dict[str, Any],
    reports: Dict[str, str],
) -> Dict[str, Any]:
    """Build a radiology request with training-equivalent text boundaries."""
    images = _sample_images(sample)
    current_image = images[0]
    prior_image = images[1] if len(images) > 1 else current_image
    study_id = _study_id(current_image)
    if study_id not in indications or not isinstance(indications[study_id], dict):
        raise KeyError(f"clinical context is missing study {study_id}")
    indication = indications[study_id].get("Indication", "")
    if _is_missing_scalar(indication):
        indication = ""

    if current_image != prior_image:
        prior_id = _study_id(prior_image)
        if prior_id not in reports:
            raise KeyError(f"prior report is missing study {prior_id}")
        prior_report = reports[prior_id]
        if indication == "":
            opening_text = f"{SYSTEM_PROMPT}\n\nPrior chest X-ray:\n"
        else:
            opening_text = (
                f"{SYSTEM_PROMPT}\n\nIndication: {indication}\nPrior chest X-ray:\n"
            )
        content = [
            {"type": "text", "text": opening_text},
            {"type": "image", "image": _mimic_image_path(prior_image)},
            {
                "type": "text",
                "text": f"\nPrior findings:{prior_report}\nCurrent chest X-ray:\n",
            },
            {"type": "image", "image": _mimic_image_path(current_image)},
            {"type": "text", "text": f"\n{MCOT_INSTRUCTION_PRIOR_PROMPT}"},
        ]
    else:
        if indication == "":
            opening_text = f"{SYSTEM_PROMPT}\n\nCurrent chest X-ray:\n"
        else:
            opening_text = (
                f"{SYSTEM_PROMPT}\n\nIndication: {indication}\nCurrent chest X-ray:\n"
            )
        content = [
            {"type": "text", "text": opening_text},
            {"type": "image", "image": _mimic_image_path(current_image)},
            {"type": "text", "text": f"\n{MCOT_INSTRUCTION_PROMPT}"},
        ]
    return {
        "role": "user",
        "content": _normalize_user_content_like_training(content),
    }


def load_external_vision_model(base_model: str, checkpoint: str) -> Any:
    """Build both vision branches and strictly restore their checkpoint weights.

    Decoder tensors in the full checkpoint are intentionally not materialized:
    this model is used only to produce global/crop image embeddings.
    """

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"External vision checkpoint does not exist: {checkpoint_path}"
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    # Every local tensor is required below, so cloning the global branch first
    # would only create a large temporary copy that is immediately overwritten.
    model = attach_local_vision_branch(model, initialize_from_global=False)
    report = load_vision_checkpoint_into_model(
        model,
        checkpoint_path,
        strict=True,
        require_local=True,
    )
    print(
        "[Info] External vision checkpoint loaded: "
        f"global={report.global_loaded}, local={report.local_loaded}, "
        f"decoder_tensors_skipped={report.ignored_tensor_keys}",
        flush=True,
    )
    model.eval()
    return model


def _true_stop_token_ids(processor: Any, vision_model: Any) -> List[int]:
    """Collect every EOS accepted by the checkpoint, preserving order."""
    stop_ids: List[int] = []
    tokenizer_eos = getattr(processor.tokenizer, "eos_token_id", None)
    if tokenizer_eos is not None:
        stop_ids.append(int(tokenizer_eos))

    generation_config = getattr(vision_model, "generation_config", None)
    configured_eos = getattr(generation_config, "eos_token_id", None)
    configured_values: Iterable[Any]
    if isinstance(configured_eos, (list, tuple, set)):
        configured_values = configured_eos
    else:
        configured_values = [configured_eos]
    for token_id in configured_values:
        if token_id is not None:
            stop_ids.append(int(token_id))

    stop_ids = list(dict.fromkeys(stop_ids))
    if not stop_ids:
        raise ValueError("No EOS token IDs are available from tokenizer/generation config.")
    return stop_ids


def _prepare_exact_inputs(
    messages: List[list],
    processor: Any,
    vision_model: Any,
) -> List[Dict[str, Any]]:
    """Prepare every sample through the exact single-sample training path."""
    prepared: List[Dict[str, Any]] = []
    total = len(messages)
    for sample_index, message in enumerate(messages, start=1):
        item = prepare_inputs_for_vllm_external(
            messages=message,
            processor=processor,
            vision_model=vision_model,
        )
        if not isinstance(item, dict):
            raise RuntimeError(
                "Training input preparation must return a dictionary, got "
                f"{type(item).__name__} for sample {sample_index}."
            )
        prompt_token_ids = item.get("prompt_token_ids")
        if prompt_token_ids is None:
            raise RuntimeError(
                "Training input preparation did not return prompt_token_ids for "
                f"sample {sample_index}."
            )
        exact_item = dict(item)
        # Reuse the authoritative IDs returned by the training function.  A
        # second prompt encode here could silently diverge from rollout state.
        exact_item["prompt_token_ids"] = [int(token_id) for token_id in prompt_token_ids]
        exact_item["anchor_image"] = select_anchor_image_path(message)
        prepared.append(exact_item)

        if sample_index % PREPARE_LOG_INTERVAL == 0 or sample_index == total:
            print(
                "[Info] Prepared training-aligned external-MM inputs for "
                f"{sample_index}/{total} samples (one sample per call).",
                flush=True,
            )
    return prepared


def _temperature_tag(temperature: float) -> str:
    scaled = temperature * 10
    if abs(scaled - round(scaled)) < 1e-8:
        return f"{int(round(scaled)):02d}"
    return str(temperature).replace("-", "m").replace(".", "p")


def _presence_penalty_tag(presence_penalty: float) -> str:
    # Keep filenames portable while preserving the numeric value, for example
    # 1.5 -> 1p5 and -0.5 -> m0p5.
    return format(presence_penalty, ".12g").replace("-", "m").replace(".", "p")


def _public_results(exact_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the old schema by default; optionally persist the exact trajectory."""
    save_trace = os.environ.get("QWENVL_INFER_SAVE_TOKEN_TRACE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    public_results: List[Dict[str, Any]] = []
    for result in exact_results:
        item = {
            "id": result.get("id"),
            "text": result.get("text", ""),
        }
        if save_trace:
            item["token_ids"] = list(result.get("token_ids", []))
            # 1 is a sampled action token; 0 is an injected observation token.
            item["action_mask"] = list(result.get("action_mask", []))
        public_results.append(item)
    return public_results


def main() -> None:
    # Frequently adjusted checkpoint settings. Environment variables still
    # take precedence when launching the script from a shell.
    BASE_MODEL = os.environ.get(
        "QWENVL_INFER_BASE_MODEL",
        "Qwen/Qwen3-VL-4B-Instruct",
    )
    VISION_CHECKPOINT = os.environ.get(
        "QWENVL_INFER_VISION_CHECKPOINT",
        "",
    )
    VLLM_CHECKPOINT = os.environ.get(
        "QWENVL_INFER_VLLM_CHECKPOINT",
        "",
    )
    OUTPUT_PREFIX = os.environ.get("QWENVL_INFER_OUTPUT_PREFIX", "grpo")
    # Keep every evaluation sample in one rollout request pool, while bounding
    # the external vision tower's incremental crop encoding batch size.
    PAUSE_IMAGE_ENCODE_BATCH_SIZE = int(
        os.environ.get("QWENVL_INFER_PAUSE_IMAGE_ENCODE_BATCH_SIZE", "32")
    )

    start_time = time.time()
    required_files = {
        "QWENVL_INFER_REPORT_PATH": REPORT_PATH,
        "QWENVL_INFER_REFERENCE_REPORT_PATH": REFERENCE_REPORT_PATH,
        "QWENVL_INFER_CLINICAL_CONTEXT_PATH": CLINICAL_CONTEXT_PATH,
    }
    for variable, raw_path in required_files.items():
        if not raw_path or not Path(raw_path).is_file():
            raise FileNotFoundError(f"{variable} must point to an existing file: {raw_path!r}")
    if not MIMIC_CXR_ROOT or not Path(MIMIC_CXR_ROOT).is_dir():
        raise FileNotFoundError(
            "QWENVL_INFER_MIMIC_CXR_ROOT must point to the MIMIC-CXR image directory"
        )
    for variable, raw_path in {
        "QWENVL_INFER_VISION_CHECKPOINT": VISION_CHECKPOINT,
        "QWENVL_INFER_VLLM_CHECKPOINT": VLLM_CHECKPOINT,
    }.items():
        if not raw_path or not Path(raw_path).is_dir():
            raise FileNotFoundError(f"{variable} must point to a checkpoint directory")
    if MAX_MODEL_LEN <= 0:
        raise ValueError("QWENVL_INFER_MAX_MODEL_LEN must be > 0.")
    if MAX_TOTAL_ACTION_TOKENS <= 0:
        raise ValueError("QWENVL_INFER_MAX_TOTAL_ACTION_TOKENS must be > 0.")
    if MAX_PAUSE_TIMES < 0:
        raise ValueError("QWENVL_INFER_MAX_PAUSE_TIMES must be >= 0.")
    if PREPARE_LOG_INTERVAL <= 0:
        raise ValueError("QWENVL_INFER_PREPARE_LOG_INTERVAL must be > 0.")
    if PAUSE_IMAGE_ENCODE_BATCH_SIZE <= 0:
        raise ValueError("QWENVL_INFER_PAUSE_IMAGE_ENCODE_BATCH_SIZE must be > 0.")

    temperature = _parse_temperature(
        os.environ.get("QWENVL_INFER_TEMPERATURES", "0.7")
    )
    top_p = float(os.environ.get("QWENVL_INFER_TOP_P", "0.8"))
    top_k = int(os.environ.get("QWENVL_INFER_TOP_K", "-1"))
    max_tokens_per_round = int(
        os.environ.get("QWENVL_INFER_MAX_TOKENS_PER_ROUND", "3072")
    )
    presence_penalty = _parse_presence_penalty(
        os.environ.get(
            "QWENVL_INFER_PRESENCE_PENALTIES",
            # Backward-compatible fallback for launch commands that set a
            # single presence-penalty value.
            os.environ.get("QWENVL_INFER_PRESENCE_PENALTY", "1.5"),
        )
    )
    if not (0.0 < top_p <= 1.0):
        raise ValueError("QWENVL_INFER_TOP_P must be in (0, 1].")
    if max_tokens_per_round <= 0:
        raise ValueError("QWENVL_INFER_MAX_TOKENS_PER_ROUND must be > 0.")

    check_vllm_mm_embed_capability()
    report_list = _load_jsonl(REPORT_PATH)
    if not report_list:
        raise ValueError("the evaluation report JSONL is empty")
    reference_report_list = (
        report_list
        if Path(REFERENCE_REPORT_PATH).resolve() == Path(REPORT_PATH).resolve()
        else _load_jsonl(REFERENCE_REPORT_PATH)
    )
    with open(CLINICAL_CONTEXT_PATH, "r", encoding="utf-8") as handle:
        indications = json.load(handle)
    if not isinstance(indications, dict):
        raise TypeError("the clinical-context file must contain a JSON object")

    reports: Dict[str, str] = {}
    for sample in reference_report_list:
        study_id = _study_id(_sample_images(sample)[0])
        report = _sample_report(sample)
        if study_id in reports and reports[study_id] != report:
            raise ValueError(f"conflicting reports for study {study_id}")
        reports[study_id] = report
    messages = [
        [data_construct(sample, indications, reports)]
        for sample in report_list
    ]

    print(f"[Info] Loading external vision model from {VISION_CHECKPOINT}", flush=True)
    vision_model = load_external_vision_model(BASE_MODEL, VISION_CHECKPOINT)
    processor = AutoProcessor.from_pretrained(
        VLLM_CHECKPOINT,
        fix_mistral_regex=True,
    )
    patch_vllm_qwen3vl_mm_embed_field_sizes()
    inputs = _prepare_exact_inputs(messages, processor, vision_model)

    tensor_parallel_size = int(
        os.environ.get("QWENVL_INFER_TP_SIZE", str(torch.cuda.device_count()))
    )
    if tensor_parallel_size <= 0:
        raise ValueError("QWENVL_INFER_TP_SIZE must be > 0 and CUDA must be available.")
    engine_seed = int(os.environ.get("QWENVL_INFER_SEED", "3407"))
    llm = LLM(
        model=VLLM_CHECKPOINT,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=float(
            os.environ.get("QWENVL_INFER_GPU_MEMORY_UTILIZATION", "0.24")
        ),
        max_num_seqs=int(os.environ.get("QWENVL_INFER_MAX_NUM_SEQS", "16")),
        max_model_len=MAX_MODEL_LEN,
        enable_mm_embeds=True,
        limit_mm_per_prompt={
            "image": int(os.environ.get("QWENVL_INFER_LIMIT_MM_IMAGE", "256"))
        },
        enable_expert_parallel=False,
        # Engine-level seeding keeps runs reproducible.  Request-level seed is
        # deliberately left unset below so manually expanded/continued samples
        # do not restart the same random stream every round.
        seed=engine_seed,
    )

    stop_token_ids = _true_stop_token_ids(processor, vision_model)
    pause_token_id = int(os.environ.get("QWENVL_INFER_PAUSE_TOKEN_ID", "151649"))
    print(
        f"[Info] pause_token_id={pause_token_id}, true_stop_token_ids={stop_token_ids}, "
        f"max_model_len={MAX_MODEL_LEN}, rollout_samples={len(inputs)}, "
        f"pause_image_encode_batch_size={PAUSE_IMAGE_ENCODE_BATCH_SIZE}",
        flush=True,
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    overwrite_outputs = os.environ.get("QWENVL_INFER_OVERWRITE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    generation_start = time.time()
    output_path = output_dir / (
        f"{OUTPUT_PREFIX}_{_temperature_tag(temperature)}"
        f"_{engine_seed}_{_presence_penalty_tag(presence_penalty)}.json"
    )
    if output_path.exists() and not overwrite_outputs:
        raise FileExistsError(
            f"refusing to overwrite {output_path}; set QWENVL_INFER_OVERWRITE=1"
        )
    print(
        "[Info] Start corrected generation with "
        f"temperature={temperature}, presence_penalty={presence_penalty}",
        flush=True,
    )
    sampling_kwargs = {
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
    sampling_params = SamplingParams(**sampling_kwargs)

    # All samples remain in one vLLM request pool; only pause-image vision
    # encoding is split into bounded chunks inside the shared rollout.
    exact_results = iterative_generate_dynamic_batch_external(
        llm=llm,
        processor=processor,
        vision_model=vision_model,
        inputs=inputs,
        messages=messages,
        report_list=report_list,
        base_params=sampling_params,
        pause_token_id=pause_token_id,
        true_stop_token_id=stop_token_ids,
        max_pause_times=MAX_PAUSE_TIMES,
        max_total_new_tokens=MAX_TOTAL_ACTION_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        pause_image_encode_batch_size=PAUSE_IMAGE_ENCODE_BATCH_SIZE,
    )
    if len(exact_results) != len(inputs):
        raise RuntimeError(
            "Full-dataset rollout returned a mismatched result count: "
            f"expected={len(inputs)}, got={len(exact_results)}."
        )
    results = _public_results(exact_results)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(
        f"Done temperature={temperature}, "
        f"presence_penalty={presence_penalty}, "
        f"generated {len(results)} samples, output={output_path}, "
        f"elapsed={time.time() - generation_start:.2f}s",
        flush=True,
    )

    print(f"Elapsed: {time.time() - start_time:.2f}s", flush=True)


if __name__ == "__main__":
    main()
