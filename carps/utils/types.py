"""Type Aliases."""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

import numpy as np

from carps.utils.trials import TrialInfo, TrialValue

SearchSpace = Any
Cost = np.ndarray | float
Incumbent = tuple[TrialInfo, TrialValue] | list[tuple[TrialInfo, TrialValue]] | None

VERY_SMALL_NUMBER = -1e32
VERY_LARGE_NUMBER = 1e32


class RunStatus(Enum):
    """Enum for the status of a run."""

    COMPLETED = auto()
    MISSING = auto()
    TRUNCATED = auto()
