"""Dataset registry configured entirely through public, machine-neutral paths."""

from __future__ import annotations

import os
import re
from pathlib import Path

from project_paths import ABNORMAL_MCOT_ANNOTATION, IMAGE_ROOT, MCOT_ANNOTATION


MIMIC_COT = {
    "annotation_path": os.environ.get(
        "MCOT_ANNOTATION_PATH", str(MCOT_ANNOTATION)
    ),
    "data_path": os.environ.get("MIMIC_CXR_ROOT", str(IMAGE_ROOT)),
}

ABN_MIMIC_COT = {
    "annotation_path": os.environ.get(
        "MCOT_ABNORMAL_ANNOTATION_PATH",
        str(ABNORMAL_MCOT_ANNOTATION),
    ),
    "data_path": os.environ.get("MIMIC_CXR_ROOT", str(IMAGE_ROOT)),
}

data_dict = {
    "MIMIC_COT": MIMIC_COT,
    "ABN_MIMIC_COT": ABN_MIMIC_COT,
    "MCOT": MIMIC_COT,
    "MCOT_ABNORMAL": ABN_MIMIC_COT,
}


def parse_sampling_rate(dataset_name: str) -> float:
    match = re.search(r"%(\d+)$", dataset_name)
    if not match:
        return 1.0
    percentage = int(match.group(1))
    if not 0 < percentage <= 100:
        raise ValueError(f"sampling percentage must be in [1, 100], got {percentage}")
    return percentage / 100.0


def data_list(dataset_names: list[str]) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for raw_name in dataset_names:
        sampling_rate = parse_sampling_rate(raw_name)
        dataset_name = re.sub(r"%(\d+)$", "", raw_name)
        if dataset_name not in data_dict:
            choices = ", ".join(sorted(data_dict))
            raise ValueError(f"unknown dataset {dataset_name!r}; available datasets: {choices}")
        config = data_dict[dataset_name].copy()
        annotation_path = Path(str(config["annotation_path"]))
        data_path = Path(str(config["data_path"])) if config["data_path"] else None
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"annotation file for {dataset_name!r} does not exist: {annotation_path}"
            )
        if data_path is None or not data_path.is_dir():
            raise FileNotFoundError(
                f"set MIMIC_CXR_ROOT to an existing image directory for {dataset_name!r}"
            )
        config["sampling_rate"] = sampling_rate
        configs.append(config)
    return configs
