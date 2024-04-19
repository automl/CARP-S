from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from ConfigSpace import Configuration

from carps.utils.trials import TrialInfo, TrialValue

SearchSpace: TypeAlias = Any
Cost: TypeAlias = np.ndarray | float
Incumbent: TypeAlias = tuple[TrialInfo, TrialValue] | list[tuple[TrialInfo, TrialValue]] | None
