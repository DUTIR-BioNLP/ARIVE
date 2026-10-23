"""Export a full ARIVE checkpoint into the decoder-only layout used by vLLM."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


LOCAL_VISUAL_PREFIX = "model.local_visual."
DROP_EXACT_KEYS = {"lm_head.weight"}
NON_INFERENCE_ARTIFACT_NAMES = {
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "scaler.pt",
    "zero_to_fp32.py",
}
NON_METADATA_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export a Qwen3-VL checkpoint for vLLM use. "
            "If the source is a single-file HF checkpoint such as checkpoint-200, "
            "the script first converts it into a safetensors+index checkpoint, then exports."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source checkpoint directory.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Destination directory for the vLLM-ready checkpoint.",
    )
    parser.add_argument(
        "--safe-dir",
        type=Path,
        default=None,
        help=(
            "Optional intermediate directory for auto-converted safetensors checkpoint. "
            "Defaults to '<src>-safe' when needed."
        ),
    )
    return parser.parse_args()


def should_drop_key(key: str) -> bool:
    return key.startswith(LOCAL_VISUAL_PREFIX) or key in DROP_EXACT_KEYS


def load_single_weight_file(weight_path: Path):
    if weight_path.suffix == ".safetensors":
        return load_file(str(weight_path), device="cpu")

    try:
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions predating weights_only.
        state_dict = torch.load(weight_path, map_location="cpu")
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint is not a state dictionary: {weight_path}")
    return state_dict


def copy_non_weight_files(src: Path, dst: Path, skip_names: set[str]):
    for path in src.iterdir():
        if not path.is_file():
            continue
        if path.name in skip_names:
            continue
        if path.name in NON_INFERENCE_ARTIFACT_NAMES or path.name.startswith("rng_state"):
            continue
        if path.suffix.lower() in NON_METADATA_SUFFIXES:
            continue
        if path.name.startswith("events.out.tfevents"):
            continue
        shutil.copy2(path, dst / path.name)


def sanitize_state_dict_for_safetensors(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    sanitized: dict[str, torch.Tensor] = {}
    seen_ptrs: dict[tuple[int, int, int], str] = {}

    for key, value in state_dict.items():
        # Keep local_visual in the intermediate safe checkpoint so it can still
        # be used as FT_CKPT by the external-vision path. Only drop known tied
        # duplicate heads such as lm_head.weight here.
        if key in DROP_EXACT_KEYS:
            continue

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state-dict value is not a tensor: key={key!r}")

        ptr_key = (value.untyped_storage().data_ptr(), value.storage_offset(), value.numel())
        if ptr_key in seen_ptrs:
            # Tied weights like lm_head / embed_tokens commonly share storage.
            # If the current key is a known throwaway key, just skip it.
            if key in DROP_EXACT_KEYS:
                continue
            value = value.clone()
        else:
            seen_ptrs[ptr_key] = key

        sanitized[key] = value.contiguous()

    return sanitized


def ensure_safe_checkpoint(src: Path, safe_dir: Path | None) -> Path:
    if not src.is_dir():
        raise FileNotFoundError(f"source checkpoint directory does not exist: {src}")
    safe_index = src / "model.safetensors.index.json"
    if safe_index.exists():
        return src

    single_weight_candidates = [
        src / "model.safetensors",
        src / "pytorch_model.bin",
    ]
    weight_path = next((path for path in single_weight_candidates if path.exists()), None)
    if weight_path is None:
        raise FileNotFoundError(
            f"Unsupported checkpoint layout under {src}: expected model.safetensors.index.json "
            f"or a single-file weight such as pytorch_model.bin."
        )

    safe_dir = safe_dir or src.with_name(f"{src.name}-safe")
    if safe_dir.exists():
        if not safe_dir.is_dir():
            raise NotADirectoryError(f"intermediate path is not a directory: {safe_dir}")
        if any(safe_dir.iterdir()):
            raise FileExistsError(f"intermediate directory must be absent or empty: {safe_dir}")
    safe_dir.mkdir(parents=True, exist_ok=True)

    copy_non_weight_files(
        src,
        safe_dir,
        skip_names={
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        },
    )

    print(f"[convert] loading single-file checkpoint from {weight_path}")
    state_dict = load_single_weight_file(weight_path)
    state_dict = sanitize_state_dict_for_safetensors(state_dict)
    out_weight = safe_dir / "model.safetensors"
    save_file(state_dict, str(out_weight))

    index = {
        "metadata": {
            "total_size": sum(value.numel() * value.element_size() for value in state_dict.values())
        },
        "weight_map": {key: out_weight.name for key in state_dict.keys()},
    }
    (safe_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2)
    )

    print(f"[convert] wrote safetensors checkpoint to {safe_dir}")
    print(f"[convert] num_keys={len(state_dict)}")
    return safe_dir


def export_vllm_checkpoint(src: Path, dst: Path):
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Expected safetensors index not found: {index_path}")
    if not (src / "config.json").is_file():
        raise FileNotFoundError(f"model config not found: {src / 'config.json'}")

    old_index = json.loads(index_path.read_text())
    old_map = old_index["weight_map"]
    shard_names = sorted({name for name in old_map.values()})

    if dst.exists():
        if not dst.is_dir():
            raise NotADirectoryError(f"destination is not a directory: {dst}")
        if any(dst.iterdir()):
            raise FileExistsError(f"destination must be absent or empty: {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    copy_non_weight_files(src, dst, skip_names=set(shard_names) | {"model.safetensors.index.json"})

    new_weight_map = {}
    total_tensor_bytes = 0
    for shard_name in shard_names:
        src_shard = src / shard_name
        dst_shard = dst / shard_name

        keep_tensors = {}
        with safe_open(src_shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if should_drop_key(key):
                    continue
                tensor = handle.get_tensor(key)
                keep_tensors[key] = tensor
                total_tensor_bytes += tensor.numel() * tensor.element_size()

        save_file(keep_tensors, str(dst_shard))

    for key, shard_name in old_map.items():
        if should_drop_key(key):
            continue
        new_weight_map[key] = shard_name

    new_index = dict(old_index)
    new_index["weight_map"] = new_weight_map
    new_index.setdefault("metadata", {})
    new_index["metadata"]["total_size"] = total_tensor_bytes

    (dst / "model.safetensors.index.json").write_text(
        json.dumps(new_index, ensure_ascii=False, indent=2)
    )

    print(f"[export] done: {dst}")
    print(f"[export] kept_keys={len(new_weight_map)}")
    print(f"[export] removed_keys={len(old_map) - len(new_weight_map)}")


def main():
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    safe_dir = args.safe_dir.resolve() if args.safe_dir is not None else None

    export_src = ensure_safe_checkpoint(src, safe_dir)
    print(f"[export] source={export_src}")
    export_vllm_checkpoint(export_src, dst)


if __name__ == "__main__":
    main()
