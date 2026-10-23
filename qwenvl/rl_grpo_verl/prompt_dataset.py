from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset

from qwenvl.data import data_list
from qwenvl.data.data_processor import _build_messages, parse_path_and_bbox, read_jsonl

from .interfaces import PromptSample


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _count_placeholder(conversations: Sequence[Dict[str, Any]], token: str) -> int:
    total = 0
    for turn in conversations:
        value = turn.get("value", "")
        total += value.count(token)
    return total


def _resolve_anchor_image(prompt_item: Dict[str, Any]) -> Optional[str]:
    image_entries = _as_list(prompt_item.get("image"))
    if not image_entries:
        return None
    raw_anchor = image_entries[-1]
    if not isinstance(raw_anchor, str) or not raw_anchor:
        return None
    anchor_path, _ = parse_path_and_bbox(raw_anchor)
    if not anchor_path:
        return None
    base_path = Path(prompt_item.get("data_path", ""))
    return str((base_path / anchor_path).resolve())


def _extract_prompt_and_target(item: Dict[str, Any]) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    conversations = item.get("conversations") or []
    if not conversations:
        return None

    last_assistant_idx = None
    for idx in range(len(conversations) - 1, -1, -1):
        role = conversations[idx].get("from", "")
        if role in ("gpt", "assistant"):
            last_assistant_idx = idx
            break
    if last_assistant_idx is None or last_assistant_idx == 0:
        return None

    prompt_turns = conversations[:last_assistant_idx]
    target_answer = conversations[last_assistant_idx].get("value", "")
    if not prompt_turns:
        return None
    return prompt_turns, target_answer


@dataclass
class PromptDatasetConfig:
    dataset_use: str
    enable_local_image: bool = False
    shuffle: bool = True
    max_prompt_samples: int = -1
    seed: int = 42


@dataclass
class _PromptRecord:
    uid: str
    prompt_item: Dict[str, Any]
    ground_truth: str
    meta: Dict[str, Any]


class GRPOPromptDataset(Dataset):
    def __init__(self, cfg: PromptDatasetConfig):
        super().__init__()
        self.cfg = cfg
        self.samples: List[_PromptRecord] = []
        self._build()

    def _build(self):
        dataset_keys = [x for x in self.cfg.dataset_use.split(",") if x]
        dataset_cfgs = data_list(dataset_keys)
        rows: List[_PromptRecord] = []
        target_samples = self.cfg.max_prompt_samples if self.cfg.max_prompt_samples > 0 else None

        for dataset_id, ds in enumerate(dataset_cfgs):
            ann_path = ds["annotation_path"]
            if ann_path.endswith(".jsonl"):
                annotations = read_jsonl(ann_path)
            else:
                with open(ann_path, "r") as f:
                    annotations = json.load(f)

            if ds.get("sampling_rate", 1.0) < 1.0:
                k = max(1, int(len(annotations) * ds["sampling_rate"]))
                annotations = random.sample(annotations, k)

            for idx, raw_ann in enumerate(annotations):
                ann_items = raw_ann if isinstance(raw_ann, list) else [raw_ann]
                for sub_idx, item in enumerate(ann_items):
                    if not isinstance(item, dict):
                        continue
                    parsed = _extract_prompt_and_target(item)
                    if parsed is None:
                        continue
                    prompt_turns, ground_truth = parsed

                    image_limit = _count_placeholder(prompt_turns, "<image>")
                    video_limit = _count_placeholder(prompt_turns, "<video>")
                    prompt_item = dict(item)
                    prompt_item["conversations"] = prompt_turns
                    prompt_item["image"] = _as_list(item.get("image"))[:image_limit]
                    prompt_item["video"] = _as_list(item.get("video"))[:video_limit]
                    prompt_item["data_path"] = ds["data_path"]

                    uid = f"{dataset_id}:{idx}:{sub_idx}"
                    rows.append(
                        _PromptRecord(
                            uid=uid,
                            prompt_item=prompt_item,
                            ground_truth=ground_truth,
                            meta={
                                "dataset_name": (
                                    dataset_keys[dataset_id] if dataset_id < len(dataset_keys) else str(dataset_id)
                                ),
                                "data_source": ds["annotation_path"],
                                "raw_item": item,
                            },
                        )
                    )
                    if target_samples is not None and not self.cfg.shuffle and len(rows) >= target_samples:
                        break
                if target_samples is not None and not self.cfg.shuffle and len(rows) >= target_samples:
                    break
            if target_samples is not None and not self.cfg.shuffle and len(rows) >= target_samples:
                break

        if self.cfg.shuffle:
            random.seed(self.cfg.seed)
            random.shuffle(rows)
        if self.cfg.max_prompt_samples > 0:
            rows = rows[: self.cfg.max_prompt_samples]
        self.samples = rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PromptSample:
        record = self.samples[idx]
        try:
            messages, image_branch_flags = _build_messages(
                record.prompt_item,
                base_path=Path(record.prompt_item.get("data_path", "")),
                enable_local_image=self.cfg.enable_local_image,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build prompt sample uid={record.uid} "
                f"from {record.meta.get('data_source', '<unknown>')}: {exc}"
            ) from exc

        return PromptSample(
            uid=record.uid,
            messages=messages,
            ground_truth=record.ground_truth,
            image_branch_flags=image_branch_flags if image_branch_flags else None,
            anchor_image=_resolve_anchor_image(record.prompt_item),
            meta=record.meta,
        )
