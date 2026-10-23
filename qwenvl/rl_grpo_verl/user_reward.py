from __future__ import annotations

from typing import Any, Dict, Optional


def compute_reward(solution_str: str, ground_truth: str, meta: Optional[Dict[str, Any]] = None) -> float:
    """
    Optional user reward function.
    Signature is compatible with `--reward_module qwenvl.rl_grpo_verl.user_reward`.
    """
    if not solution_str or not solution_str.strip():
        return -1.0
    if ground_truth and solution_str.strip() == ground_truth.strip():
        return 1.0
    return 0.0

