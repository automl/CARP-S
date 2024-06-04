from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from carps.utils.trials import TrialInfo, TrialValue

SearchSpace = Any
Cost = Union[np.ndarray, float]
Incumbent = Optional[tuple[TrialInfo, TrialValue] | list[tuple[TrialInfo, TrialValue]]]
