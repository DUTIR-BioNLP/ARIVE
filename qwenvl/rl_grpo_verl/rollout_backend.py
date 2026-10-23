from __future__ import annotations

import abc
import importlib
from typing import List

from .interfaces import RolloutRequest, RolloutResponse


class BaseRolloutBackend(abc.ABC):
    """User-implemented rollout inference backend."""

    @abc.abstractmethod
    def generate(self, requests: List[RolloutRequest]) -> List[RolloutResponse]:
        raise NotImplementedError


def load_rollout_backend(module_name: str, class_name: str) -> type[BaseRolloutBackend]:
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not issubclass(cls, BaseRolloutBackend):
        raise TypeError(f"{module_name}.{class_name} must inherit BaseRolloutBackend")
    return cls

