from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from qwenvl.train.checkpoint_loader import (
    inspect_vision_checkpoint,
    load_checkpoint_into_model,
    load_vision_checkpoint_into_model,
    resolve_checkpoint_layout,
)


class TinyVision(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(3, 2)
        self.register_buffer("scale", torch.zeros(2))


class TinyCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = TinyVision()
        self.local_visual = TinyVision()
        self.language_model = torch.nn.Linear(2, 2)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyCore()


def _filled_checkpoint_state(model: TinyModel) -> dict[str, torch.Tensor]:
    state = {}
    for key, value in model.state_dict().items():
        if key.startswith("model.visual."):
            fill = 1.0
        elif key.startswith("model.local_visual."):
            fill = 2.0
        else:
            fill = 3.0
        state[key] = torch.full_like(value, fill).contiguous()
    return state


def _write_sharded_checkpoint(
    root: Path,
    state: dict[str, torch.Tensor],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    first = {
        key: value
        for key, value in state.items()
        if not key.startswith("model.local_visual.")
    }
    second = {
        key: value
        for key, value in state.items()
        if key.startswith("model.local_visual.")
    }
    shard_names = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    save_file(first, str(root / shard_names[0]))
    save_file(second, str(root / shard_names[1]))
    weight_map = {
        **{key: shard_names[0] for key in first},
        **{key: shard_names[1] for key in second},
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )


class CheckpointLoaderTest(unittest.TestCase):
    def test_selective_vision_load_skips_decoder_and_restores_both_branches(
        self,
    ) -> None:
        source_model = TinyModel()
        state = _filled_checkpoint_state(source_model)
        target_model = TinyModel()
        for value in target_model.state_dict().values():
            value.zero_()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint"
            _write_sharded_checkpoint(checkpoint, state)

            inspection = inspect_vision_checkpoint(checkpoint)
            self.assertEqual(inspection.global_keys, 3)
            self.assertEqual(inspection.local_keys, 3)
            self.assertEqual(inspection.shape_pairs_checked, 3)

            report = load_vision_checkpoint_into_model(target_model, checkpoint)

        self.assertEqual(report.global_loaded, 3)
        self.assertEqual(report.local_loaded, 3)
        self.assertEqual(report.ignored_tensor_keys, 2)
        for key, value in target_model.state_dict().items():
            if key.startswith("model.visual."):
                self.assertTrue(torch.equal(value, torch.ones_like(value)), key)
            elif key.startswith("model.local_visual."):
                self.assertTrue(torch.equal(value, torch.full_like(value, 2)), key)
            else:
                self.assertTrue(torch.equal(value, torch.zeros_like(value)), key)

    def test_full_overlay_streams_all_model_tensors(self) -> None:
        source_model = TinyModel()
        state = _filled_checkpoint_state(source_model)
        target_model = TinyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint"
            _write_sharded_checkpoint(checkpoint, state)
            report = load_checkpoint_into_model(
                target_model,
                checkpoint,
                scope="all",
                strict=True,
                require_local=True,
            )

        self.assertEqual(report.loaded_tensor_keys, len(state))
        for key, value in target_model.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]), key)

    def test_missing_local_tensor_fails_before_loading(self) -> None:
        model = TinyModel()
        state = _filled_checkpoint_state(model)
        state.pop("model.local_visual.proj.bias")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint"
            _write_sharded_checkpoint(checkpoint, state)
            with self.assertRaisesRegex(RuntimeError, "key sets do not match"):
                inspect_vision_checkpoint(checkpoint)
            with self.assertRaisesRegex(RuntimeError, "Strong local-vision validation"):
                load_vision_checkpoint_into_model(model, checkpoint)

    def test_local_shape_mismatch_is_rejected(self) -> None:
        model = TinyModel()
        state = _filled_checkpoint_state(model)
        state["model.local_visual.proj.weight"] = torch.zeros(4, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint"
            _write_sharded_checkpoint(checkpoint, state)
            with self.assertRaisesRegex(RuntimeError, "shapes do not match"):
                inspect_vision_checkpoint(checkpoint)
            with self.assertRaisesRegex(RuntimeError, "Strong local-vision validation"):
                load_vision_checkpoint_into_model(model, checkpoint)

    def test_index_cannot_reference_a_shard_outside_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            outside = root / "outside.safetensors"
            save_file({"model.visual.weight": torch.ones(1)}, str(outside))
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {"model.visual.weight": "../outside.safetensors"}}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_checkpoint_layout(checkpoint)


if __name__ == "__main__":
    unittest.main()
