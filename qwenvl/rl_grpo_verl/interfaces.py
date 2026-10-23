from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptSample:
    uid: str
    messages: List[Dict[str, Any]]
    ground_truth: str
    meta: Dict[str, Any] = field(default_factory=dict)
    image_branch_flags: Optional[List[int]] = None
    anchor_image: Optional[str] = None


@dataclass
class RolloutRequest:
    uid: str
    messages: List[Dict[str, Any]]
    num_generations: int
    max_new_tokens: int
    temperature: float
    top_p: float
    meta: Dict[str, Any] = field(default_factory=dict)
    image_branch_flags: Optional[List[int]] = None
    anchor_image: Optional[str] = None


@dataclass
class Completion:
    text: str
    # Exact assistant-content trajectory used by the rollout policy.  For
    # interactive rollouts this can also contain environment observation
    # tokens; ``action_mask`` distinguishes sampled actions from observations.
    token_ids: Optional[List[int]] = None
    action_mask: Optional[List[int]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResponse:
    uid: str
    completions: List[Completion]
    anchor_image: Optional[str] = None
