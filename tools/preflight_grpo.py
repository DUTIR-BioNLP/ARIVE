"""Validate ARIVE GRPO artifacts and runtime without loading model weights."""

from __future__ import annotations

import argparse
import importlib
import json
import socket
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwenvl.data.data_processor import update_processor_pixels  # noqa: E402
from qwenvl.rl_grpo_verl.user_rollout import UserRolloutBackend  # noqa: E402
from qwenvl.train.argument import DataArguments  # noqa: E402
from qwenvl.train.checkpoint_loader import inspect_vision_checkpoint  # noqa: E402


WEIGHT_FILES = (
    "model.safetensors.index.json",
    "model.safetensors",
    "pytorch_model.bin.index.json",
    "pytorch_model.bin",
)
EXPECTED_TOKEN_IDS = {
    "<|endoftext|>": 151643,
    "<|im_end|>": 151645,
    "<|box_end|>": 151649,
    "<think>": 151667,
    "</think>": 151668,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="Policy base path or model id.")
    parser.add_argument("--reference", required=True, help="Reference base path or model id.")
    parser.add_argument("--overlay", type=Path, required=True, help="Full SFT checkpoint.")
    parser.add_argument("--vllm", type=Path, required=True, help="vLLM checkpoint export.")
    parser.add_argument("--visible-gpu-count", type=int, required=True)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--vllm-port", type=int, required=True)
    return parser.parse_args()


def validate_index(index_path: Path) -> None:
    with index_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"empty or invalid weight_map: {index_path}")
    if not all(isinstance(name, str) and name for name in weight_map.values()):
        raise TypeError(f"weight_map contains invalid shard names: {index_path}")
    missing = sorted(
        shard_name
        for shard_name in set(weight_map.values())
        if not (index_path.parent / shard_name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"missing shards under {index_path.parent}: {missing}")


def validate_local_checkpoint(
    path: Path,
    *,
    role: str,
    require_config: bool,
    require_preprocessor: bool,
) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{role} directory does not exist: {path}")
    if require_config and not (path / "config.json").is_file():
        raise FileNotFoundError(f"{role} is missing config.json: {path}")
    if require_preprocessor and not (path / "preprocessor_config.json").is_file():
        raise FileNotFoundError(f"{role} is missing preprocessor_config.json: {path}")
    present_weights = [name for name in WEIGHT_FILES if (path / name).is_file()]
    if not present_weights:
        raise FileNotFoundError(f"{role} contains no supported Hugging Face weights: {path}")
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if index_path.is_file():
            validate_index(index_path)


def validate_model_reference(model: str, *, role: str) -> None:
    path = Path(model).expanduser()
    if path.exists():
        validate_local_checkpoint(
            path.resolve(),
            role=role,
            require_config=True,
            require_preprocessor=False,
        )
    AutoConfig.from_pretrained(model)


def validate_ports(host: str, ports: list[int]) -> None:
    for port in ports:
        if not 1 <= port <= 65535:
            raise ValueError(f"port is outside [1, 65535]: {port}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError as exc:
                raise OSError(f"cannot bind {host}:{port}: {exc}") from None


def validate_processors(policy: str, rollout_path: Path) -> None:
    policy_processor = AutoProcessor.from_pretrained(policy, fix_mistral_regex=True)
    data_args = DataArguments()
    data_args.min_pixels = 784
    data_args.max_pixels = 50176
    policy_processor = update_processor_pixels(policy_processor, data_args)
    rollout_processor = AutoProcessor.from_pretrained(
        str(rollout_path),
        fix_mistral_regex=True,
        local_files_only=True,
    )

    validator = object.__new__(UserRolloutBackend)
    validator.policy_processor = policy_processor
    validator.processor = rollout_processor
    validator._validate_rollout_processor_matches_policy()

    tokenizer = policy_processor.tokenizer
    mismatches = {
        token: (expected, tokenizer.convert_tokens_to_ids(token))
        for token, expected in EXPECTED_TOKEN_IDS.items()
        if tokenizer.convert_tokens_to_ids(token) != expected
    }
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    if assistant_ids != [77091]:
        mismatches["assistant"] = ([77091], assistant_ids)
    if mismatches:
        raise ValueError(f"policy tokenizer does not match ARIVE token constants: {mismatches}")


def validate_runtime_packages() -> None:
    for package in ("deepspeed", "flash_attn", "verl", "vllm"):
        importlib.import_module(package)
    if Qwen3VLForConditionalGeneration is None:  # pragma: no cover
        raise ImportError("Transformers does not provide Qwen3-VL")


def main() -> None:
    args = parse_args()
    if args.visible_gpu_count < 4:
        raise ValueError("the released reward layout requires at least four visible GPUs")
    if not 1 <= args.nproc <= args.visible_gpu_count:
        raise ValueError("nproc must lie in [1, visible_gpu_count]")
    if not 1 <= args.tp_size <= args.visible_gpu_count:
        raise ValueError("vLLM tensor parallel size must lie in [1, visible_gpu_count]")
    if args.nproc > 1 and args.tp_size == 1:
        raise ValueError("multi-rank shared rollout requires vLLM tensor parallel size > 1")
    if args.master_port == args.vllm_port:
        raise ValueError("torchrun and vLLM rendezvous ports must differ")

    actual_gpu_count = torch.cuda.device_count()
    if not torch.cuda.is_available() or actual_gpu_count != args.visible_gpu_count:
        raise RuntimeError(
            "CUDA visibility mismatch: "
            f"launcher_count={args.visible_gpu_count}, torch_count={actual_gpu_count}"
        )

    overlay = args.overlay.expanduser().resolve()
    rollout = args.vllm.expanduser().resolve()
    validate_local_checkpoint(
        overlay,
        role="full SFT overlay",
        require_config=True,
        require_preprocessor=False,
    )
    vision_inspection = inspect_vision_checkpoint(
        overlay,
        require_global=True,
        require_local=True,
    )
    print(f"[grpo] vision checkpoint validated: {vision_inspection.summary()}")
    validate_local_checkpoint(
        rollout,
        role="vLLM export",
        require_config=True,
        require_preprocessor=True,
    )
    if not (rollout / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("vLLM export must contain model.safetensors.index.json")

    validate_runtime_packages()
    validate_model_reference(args.policy, role="policy base model")
    if args.reference != args.policy:
        validate_model_reference(args.reference, role="reference base model")
    validate_processors(args.policy, rollout)
    validate_ports(args.master_addr, [args.master_port, args.vllm_port])
    print("[grpo] artifact/runtime preflight passed")


if __name__ == "__main__":
    main()
