from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from carps.utils.trials import TrialInfo, TrialValue

SearchSpace = Any
Cost = Union[np.ndarray, float]
Incumbent = Optional[
    Union[Tuple[TrialInfo, TrialValue], List[Tuple[TrialInfo, TrialValue]]]
]
