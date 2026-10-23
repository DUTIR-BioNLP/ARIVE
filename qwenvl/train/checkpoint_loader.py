"""Memory-bounded loading and validation for ARIVE model checkpoints.

The released training pipeline stores a standard Qwen3-VL branch under
``model.visual.*`` and the crop branch under ``model.local_visual.*``.  Hugging
Face does not know about the latter, so it cannot be restored by an ordinary
``from_pretrained`` call.  This module provides two deliberately separate
loading modes:

* ``scope="all"`` overlays a complete SFT/GRPO checkpoint onto a model;
* ``scope="vision"`` reads only the global and local visual tensors.

Indexed safetensors checkpoints are opened lazily and only requested tensors
are materialized.  PyTorch shards are handled one shard at a time.  This keeps
peak host memory bounded and lets inference avoid reading decoder weights when
it only needs the external visual encoder.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Literal, Mapping, Optional

import torch


GLOBAL_VISUAL_PREFIX = "model.visual."
LOCAL_VISUAL_PREFIX = "model.local_visual."
VISION_PREFIXES = (GLOBAL_VISUAL_PREFIX, LOCAL_VISUAL_PREFIX)


@dataclass(frozen=True)
class CheckpointLayout:
    source: Path
    format: str
    weight_map: Optional[Dict[str, Path]]
    single_torch_file: Optional[Path] = None

    @property
    def tensor_keys(self) -> Optional[set[str]]:
        if self.weight_map is None:
            return None
        return set(self.weight_map)


@dataclass(frozen=True)
class VisionCheckpointInspection:
    source: Path
    format: str
    global_keys: int
    local_keys: int
    shape_pairs_checked: int
    deferred: bool = False

    def summary(self) -> str:
        if self.deferred:
            return (
                f"source={self.source}, format={self.format}, "
                "key/shape validation=deferred until weight loading"
            )
        return (
            f"source={self.source}, format={self.format}, "
            f"global_keys={self.global_keys}, local_keys={self.local_keys}, "
            f"shape_pairs_checked={self.shape_pairs_checked}"
        )


@dataclass(frozen=True)
class CheckpointLoadReport:
    source: Path
    format: str
    scope: str
    checkpoint_tensor_keys: int
    selected_tensor_keys: int
    loaded_tensor_keys: int
    ignored_tensor_keys: int
    global_loaded: int
    local_loaded: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]

    def summary(self) -> str:
        return (
            f"source={self.source}, format={self.format}, scope={self.scope}, "
            f"checkpoint_keys={self.checkpoint_tensor_keys}, "
            f"selected_keys={self.selected_tensor_keys}, "
            f"loaded_keys={self.loaded_tensor_keys}, "
            f"global_loaded={self.global_loaded}, local_loaded={self.local_loaded}, "
            f"ignored_keys={self.ignored_tensor_keys}, "
            f"missing={len(self.missing_keys)}, "
            f"unexpected={len(self.unexpected_keys)}, "
            f"shape_mismatch={len(self.shape_mismatches)}"
        )


def _is_main_process() -> bool:
    return not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    )


def _checkpoint_error(
    message: str,
    *,
    missing: Iterable[str] = (),
    unexpected: Iterable[str] = (),
    shape_mismatches: Iterable[tuple[str, tuple[int, ...], tuple[int, ...]]] = (),
) -> RuntimeError:
    missing = tuple(missing)
    unexpected = tuple(unexpected)
    shape_mismatches = tuple(shape_mismatches)
    details = []
    if missing:
        details.append(f"missing sample={list(missing[:5])}")
    if unexpected:
        details.append(f"unexpected sample={list(unexpected[:5])}")
    if shape_mismatches:
        details.append(f"shape mismatch sample={list(shape_mismatches[:5])}")
    suffix = f" ({'; '.join(details)})" if details else ""
    return RuntimeError(message + suffix)


def _resolve_index(index_path: Path, *, expected_suffix: str) -> CheckpointLayout:
    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read checkpoint index {index_path}: {exc}") from exc

    raw_weight_map = index_data.get("weight_map")
    if not isinstance(raw_weight_map, dict) or not raw_weight_map:
        raise ValueError(f"Checkpoint index has no non-empty weight_map: {index_path}")

    root = index_path.parent.resolve()
    weight_map: Dict[str, Path] = {}
    for key, shard_name in raw_weight_map.items():
        if not isinstance(key, str) or not key:
            raise TypeError(
                f"Checkpoint index contains an invalid tensor key: {index_path}"
            )
        if not isinstance(shard_name, str) or not shard_name:
            raise TypeError(
                f"Checkpoint index contains an invalid shard for key {key!r}: {index_path}"
            )
        shard_path = (root / shard_name).resolve()
        try:
            shard_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Checkpoint shard escapes its checkpoint directory: {shard_name!r}"
            ) from exc
        if shard_path.suffix != expected_suffix:
            raise ValueError(
                f"Checkpoint index {index_path.name} references an unexpected shard "
                f"format: {shard_path.name}"
            )
        if not shard_path.is_file():
            raise FileNotFoundError(f"Checkpoint shard does not exist: {shard_path}")
        weight_map[key] = shard_path

    return CheckpointLayout(
        source=index_path.parent.resolve(),
        format=(
            "safetensors-index"
            if expected_suffix == ".safetensors"
            else "pytorch-index"
        ),
        weight_map=weight_map,
    )


def _safetensor_keys(path: Path) -> set[str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return set(handle.keys())


def resolve_checkpoint_layout(model_path: str | Path) -> CheckpointLayout:
    """Resolve one supported local Hugging Face checkpoint layout."""

    path = Path(model_path).expanduser().resolve()
    if path.is_file():
        if path.name == "model.safetensors.index.json":
            return _resolve_index(path, expected_suffix=".safetensors")
        if path.name == "pytorch_model.bin.index.json":
            return _resolve_index(path, expected_suffix=".bin")
        if path.suffix == ".safetensors":
            keys = _safetensor_keys(path)
            if not keys:
                raise ValueError(f"Safetensors checkpoint contains no tensors: {path}")
            return CheckpointLayout(
                source=path,
                format="safetensors-single",
                weight_map={key: path for key in keys},
            )
        if path.suffix in {".bin", ".pt", ".pth", ".ckpt"}:
            return CheckpointLayout(
                source=path,
                format="pytorch-single",
                weight_map=None,
                single_torch_file=path,
            )
        raise ValueError(f"Unsupported checkpoint file: {path}")

    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    safe_index = path / "model.safetensors.index.json"
    if safe_index.is_file():
        return _resolve_index(safe_index, expected_suffix=".safetensors")
    torch_index = path / "pytorch_model.bin.index.json"
    if torch_index.is_file():
        return _resolve_index(torch_index, expected_suffix=".bin")

    safe_file = path / "model.safetensors"
    if safe_file.is_file():
        keys = _safetensor_keys(safe_file)
        if not keys:
            raise ValueError(f"Safetensors checkpoint contains no tensors: {safe_file}")
        return CheckpointLayout(
            source=path,
            format="safetensors-single",
            weight_map={key: safe_file for key in keys},
        )

    torch_file = path / "pytorch_model.bin"
    if torch_file.is_file():
        return CheckpointLayout(
            source=path,
            format="pytorch-single",
            weight_map=None,
            single_torch_file=torch_file,
        )

    raise FileNotFoundError(
        "No supported Hugging Face weights found under "
        f"{path}; expected a model.safetensors or pytorch_model.bin layout."
    )


def _load_torch_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    try:
        value = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old PyTorch compatibility
        value = torch.load(str(path), map_location="cpu")

    if isinstance(value, dict) and isinstance(value.get("state_dict"), dict):
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise TypeError(f"PyTorch checkpoint is not a state dictionary: {path}")

    tensors = {key: tensor for key, tensor in value.items() if torch.is_tensor(tensor)}
    if not tensors:
        raise ValueError(f"PyTorch checkpoint contains no tensor entries: {path}")
    return tensors


def _safetensor_metadata(
    layout: CheckpointLayout,
    keys: Iterable[str],
) -> tuple[Dict[str, tuple[int, ...]], Dict[str, str], bool]:
    """Read shape/dtype metadata without materializing tensor payloads."""

    from safetensors import safe_open

    if layout.weight_map is None:
        return {}, {}, False
    requested = set(keys)
    by_shard: dict[Path, list[str]] = defaultdict(list)
    for key in requested:
        by_shard[layout.weight_map[key]].append(key)

    shapes: Dict[str, tuple[int, ...]] = {}
    dtypes: Dict[str, str] = {}
    for shard_path, shard_keys in by_shard.items():
        if shard_path.suffix != ".safetensors":
            return {}, {}, False
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            missing_from_shard = sorted(set(shard_keys) - available)
            if missing_from_shard:
                raise _checkpoint_error(
                    f"Checkpoint index points to tensors absent from {shard_path}",
                    missing=missing_from_shard,
                )
            for key in shard_keys:
                tensor_slice = handle.get_slice(key)
                shapes[key] = tuple(tensor_slice.get_shape())
                dtypes[key] = str(tensor_slice.get_dtype())
    return shapes, dtypes, True


def inspect_vision_checkpoint(
    model_path: str | Path,
    *,
    require_global: bool = True,
    require_local: bool = True,
) -> VisionCheckpointInspection:
    """Validate branch completeness without loading safetensor payloads.

    For an unindexed, single-file PyTorch checkpoint the key inspection is
    deferred to :func:`load_checkpoint_into_model`, because PyTorch has no
    header-only key API.
    """

    layout = resolve_checkpoint_layout(model_path)
    checkpoint_keys = layout.tensor_keys
    if checkpoint_keys is None:
        return VisionCheckpointInspection(
            source=layout.source,
            format=layout.format,
            global_keys=0,
            local_keys=0,
            shape_pairs_checked=0,
            deferred=True,
        )

    global_keys = {
        key for key in checkpoint_keys if key.startswith(GLOBAL_VISUAL_PREFIX)
    }
    local_keys = {key for key in checkpoint_keys if key.startswith(LOCAL_VISUAL_PREFIX)}
    if require_global and not global_keys:
        raise RuntimeError(
            f"Checkpoint has no {GLOBAL_VISUAL_PREFIX} tensors: {layout.source}"
        )
    if require_local and not local_keys:
        raise RuntimeError(
            f"Checkpoint has no {LOCAL_VISUAL_PREFIX} tensors; this is likely a "
            f"vLLM export rather than a full ARIVE checkpoint: {layout.source}"
        )

    if global_keys and local_keys:
        global_suffixes = {key[len(GLOBAL_VISUAL_PREFIX) :] for key in global_keys}
        local_suffixes = {key[len(LOCAL_VISUAL_PREFIX) :] for key in local_keys}
        missing_local_suffixes = sorted(global_suffixes - local_suffixes)
        unexpected_local_suffixes = sorted(local_suffixes - global_suffixes)
        if missing_local_suffixes or unexpected_local_suffixes:
            raise _checkpoint_error(
                "Global/local vision checkpoint key sets do not match",
                missing=[
                    LOCAL_VISUAL_PREFIX + suffix for suffix in missing_local_suffixes
                ],
                unexpected=[
                    LOCAL_VISUAL_PREFIX + suffix for suffix in unexpected_local_suffixes
                ],
            )

        branch_keys = global_keys | local_keys
        shapes, dtypes, metadata_complete = _safetensor_metadata(layout, branch_keys)
        if metadata_complete:
            mismatches = []
            dtype_mismatches = []
            for suffix in sorted(global_suffixes):
                global_key = GLOBAL_VISUAL_PREFIX + suffix
                local_key = LOCAL_VISUAL_PREFIX + suffix
                if shapes[global_key] != shapes[local_key]:
                    mismatches.append(
                        (local_key, shapes[local_key], shapes[global_key])
                    )
                if dtypes[global_key] != dtypes[local_key]:
                    dtype_mismatches.append(
                        (local_key, dtypes[local_key], dtypes[global_key])
                    )
            if mismatches:
                raise _checkpoint_error(
                    "Global/local vision checkpoint shapes do not match",
                    shape_mismatches=mismatches,
                )
            if dtype_mismatches:
                raise RuntimeError(
                    "Global/local vision checkpoint dtypes do not match "
                    f"(sample={dtype_mismatches[:5]})"
                )
            shape_pairs_checked = len(global_suffixes)
        else:
            shape_pairs_checked = 0
    else:
        shape_pairs_checked = 0

    return VisionCheckpointInspection(
        source=layout.source,
        format=layout.format,
        global_keys=len(global_keys),
        local_keys=len(local_keys),
        shape_pairs_checked=shape_pairs_checked,
    )


def _iter_known_checkpoint_tensors(
    layout: CheckpointLayout,
    requested_keys: set[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    assert layout.weight_map is not None
    by_shard: dict[Path, list[str]] = defaultdict(list)
    for key in requested_keys:
        by_shard[layout.weight_map[key]].append(key)

    for shard_path in sorted(by_shard, key=lambda path: str(path)):
        shard_keys = sorted(by_shard[shard_path])
        if shard_path.suffix == ".safetensors":
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                missing_from_shard = sorted(set(shard_keys) - available)
                if missing_from_shard:
                    raise _checkpoint_error(
                        f"Checkpoint index points to tensors absent from {shard_path}",
                        missing=missing_from_shard,
                    )
                for key in shard_keys:
                    yield key, handle.get_tensor(key)
        else:
            shard_state = _load_torch_state_dict(shard_path)
            missing_from_shard = sorted(set(shard_keys) - set(shard_state))
            if missing_from_shard:
                raise _checkpoint_error(
                    f"Checkpoint index points to tensors absent from {shard_path}",
                    missing=missing_from_shard,
                )
            for key in shard_keys:
                yield key, shard_state[key]
            del shard_state


def _logical_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    ds_shape = getattr(tensor, "ds_shape", None)
    return tuple(ds_shape) if ds_shape is not None else tuple(tensor.shape)


def _copy_parameter(parameter: torch.nn.Parameter, tensor: torch.Tensor) -> None:
    expected_shape = _logical_shape(parameter)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Cannot copy tensor with shape {tuple(tensor.shape)} into parameter "
            f"with logical shape {expected_shape}"
        )

    if hasattr(parameter, "ds_id") or getattr(parameter, "ds_shape", None) is not None:
        try:
            import deepspeed
        except ImportError as exc:  # pragma: no cover - inconsistent runtime
            raise RuntimeError(
                "Found a ZeRO-partitioned parameter but DeepSpeed is unavailable"
            ) from exc
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        rank = torch.distributed.get_rank() if distributed else 0
        with deepspeed.zero.GatheredParameters([parameter], modifier_rank=0):
            if rank == 0:
                with torch.no_grad():
                    parameter.copy_(
                        tensor.to(device=parameter.device, dtype=parameter.dtype)
                    )
        return

    if parameter.device.type == "meta":
        raise RuntimeError("Cannot overlay a checkpoint onto a meta-device parameter")
    with torch.no_grad():
        parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def _copy_buffer(buffer: torch.Tensor, tensor: torch.Tensor) -> None:
    if tuple(buffer.shape) != tuple(tensor.shape):
        raise ValueError(
            f"Cannot copy tensor with shape {tuple(tensor.shape)} into buffer "
            f"with shape {tuple(buffer.shape)}"
        )
    if buffer.device.type == "meta":
        raise RuntimeError("Cannot overlay a checkpoint onto a meta-device buffer")
    with torch.no_grad():
        buffer.copy_(tensor.to(device=buffer.device, dtype=buffer.dtype))


def _matches_tied_key(model: torch.nn.Module, key: str) -> bool:
    patterns = getattr(model, "_tied_weights_keys", ()) or ()
    if isinstance(patterns, Mapping):
        patterns = patterns.keys()
    for pattern in patterns:
        if key == pattern:
            return True
        try:
            if re.fullmatch(str(pattern), key):
                return True
        except re.error:
            continue
    return False


def load_checkpoint_into_model(
    model: torch.nn.Module,
    model_path: str | Path,
    *,
    scope: Literal["all", "vision"] = "all",
    strict: bool = True,
    require_local: bool = False,
) -> CheckpointLoadReport:
    """Overlay checkpoint tensors onto ``model`` with bounded host memory.

    ``require_local`` is stronger than ordinary non-strict loading: every
    expected ``model.local_visual.*`` tensor must exist and have the exact
    logical shape even when ``strict=False`` for unrelated decoder keys.
    """

    if scope not in {"all", "vision"}:
        raise ValueError(f"Unsupported checkpoint loading scope: {scope!r}")

    layout = resolve_checkpoint_layout(model_path)
    materialized_state: Optional[Dict[str, torch.Tensor]] = None
    checkpoint_keys = layout.tensor_keys
    if checkpoint_keys is None:
        assert layout.single_torch_file is not None
        materialized_state = _load_torch_state_dict(layout.single_torch_file)
        checkpoint_keys = set(materialized_state)

    persistent_state_keys = set(model.state_dict())
    named_parameters = {
        key: value
        for key, value in model.named_parameters()
        if key in persistent_state_keys
    }
    named_buffers = {
        key: value
        for key, value in model.named_buffers()
        if key in persistent_state_keys
    }
    targets: Dict[str, torch.Tensor] = {**named_parameters, **named_buffers}
    if scope == "vision":
        expected_keys = {key for key in targets if key.startswith(VISION_PREFIXES)}
        selected_checkpoint_keys = {
            key for key in checkpoint_keys if key.startswith(VISION_PREFIXES)
        }
    else:
        expected_keys = set(targets)
        selected_checkpoint_keys = set(checkpoint_keys)

    local_expected = {
        key for key in expected_keys if key.startswith(LOCAL_VISUAL_PREFIX)
    }
    local_checkpoint = {
        key for key in selected_checkpoint_keys if key.startswith(LOCAL_VISUAL_PREFIX)
    }
    if require_local and not local_expected:
        raise RuntimeError(
            "Strong local-vision loading was requested, but the model has no "
            f"{LOCAL_VISUAL_PREFIX} branch"
        )

    missing = sorted(expected_keys - selected_checkpoint_keys)
    unexpected = sorted(selected_checkpoint_keys - expected_keys)
    if scope == "all":
        missing = [key for key in missing if not _matches_tied_key(model, key)]
        unexpected = [key for key in unexpected if not _matches_tied_key(model, key)]

    candidate_keys = expected_keys & selected_checkpoint_keys
    metadata_shapes, _, metadata_complete = _safetensor_metadata(layout, candidate_keys)
    shape_mismatches = []
    if metadata_complete:
        for key in sorted(candidate_keys):
            expected_shape = _logical_shape(targets[key])
            checkpoint_shape = metadata_shapes[key]
            if checkpoint_shape != expected_shape:
                shape_mismatches.append((key, checkpoint_shape, expected_shape))

    local_missing = sorted(local_expected - local_checkpoint)
    local_unexpected = sorted(local_checkpoint - local_expected)
    local_shape_mismatches = [
        mismatch
        for mismatch in shape_mismatches
        if mismatch[0].startswith(LOCAL_VISUAL_PREFIX)
    ]
    if require_local and (local_missing or local_unexpected or local_shape_mismatches):
        raise _checkpoint_error(
            f"Strong local-vision validation failed for {layout.source}",
            missing=local_missing,
            unexpected=local_unexpected,
            shape_mismatches=local_shape_mismatches,
        )
    if strict and (missing or unexpected or shape_mismatches):
        raise _checkpoint_error(
            f"Strict checkpoint validation failed for {layout.source}",
            missing=missing,
            unexpected=unexpected,
            shape_mismatches=shape_mismatches,
        )

    mismatched_keys = {key for key, _, _ in shape_mismatches}
    requested_keys = candidate_keys - mismatched_keys
    if not requested_keys:
        raise RuntimeError(
            f"No compatible {scope} tensors were found in checkpoint {layout.source}"
        )

    if materialized_state is None:
        tensor_iterator = _iter_known_checkpoint_tensors(layout, requested_keys)
    else:
        tensor_iterator = (
            (key, materialized_state[key]) for key in sorted(requested_keys)
        )

    loaded_keys = set()
    runtime_shape_mismatches = []
    for key, tensor in tensor_iterator:
        expected_shape = _logical_shape(targets[key])
        checkpoint_shape = tuple(tensor.shape)
        if checkpoint_shape != expected_shape:
            runtime_shape_mismatches.append((key, checkpoint_shape, expected_shape))
            continue
        if key in named_parameters:
            _copy_parameter(named_parameters[key], tensor)
        else:
            _copy_buffer(named_buffers[key], tensor)
        loaded_keys.add(key)

    del materialized_state
    if runtime_shape_mismatches:
        shape_mismatches.extend(runtime_shape_mismatches)
        runtime_local_mismatches = [
            mismatch
            for mismatch in runtime_shape_mismatches
            if mismatch[0].startswith(LOCAL_VISUAL_PREFIX)
        ]
        if strict or (require_local and runtime_local_mismatches):
            raise _checkpoint_error(
                f"Checkpoint tensor shape validation failed for {layout.source}",
                shape_mismatches=runtime_shape_mismatches,
            )

    not_loaded = sorted(requested_keys - loaded_keys)
    if not_loaded:
        raise _checkpoint_error(
            f"Checkpoint tensors were selected but not loaded from {layout.source}",
            missing=not_loaded,
        )

    report = CheckpointLoadReport(
        source=layout.source,
        format=layout.format,
        scope=scope,
        checkpoint_tensor_keys=len(checkpoint_keys),
        selected_tensor_keys=len(selected_checkpoint_keys),
        loaded_tensor_keys=len(loaded_keys),
        ignored_tensor_keys=len(checkpoint_keys - selected_checkpoint_keys),
        global_loaded=sum(key.startswith(GLOBAL_VISUAL_PREFIX) for key in loaded_keys),
        local_loaded=sum(key.startswith(LOCAL_VISUAL_PREFIX) for key in loaded_keys),
        missing_keys=tuple(missing),
        unexpected_keys=tuple(unexpected),
        shape_mismatches=tuple(shape_mismatches),
    )
    if _is_main_process():
        print(f"[CheckpointLoad] {report.summary()}", flush=True)
    return report


def load_vision_checkpoint_into_model(
    model: torch.nn.Module,
    model_path: str | Path,
    *,
    strict: bool = True,
    require_local: bool = True,
) -> CheckpointLoadReport:
    """Load only global/local vision branches from a full ARIVE checkpoint."""

    return load_checkpoint_into_model(
        model,
        model_path,
        scope="vision",
        strict=strict,
        require_local=require_local,
    )
