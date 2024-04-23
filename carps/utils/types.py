from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from ConfigSpace import Configuration

SearchSpace = Any
Cost = Union[np.ndarray, float]
Incumbent = Optional[
    Union[Tuple[Configuration, Cost], List[Tuple[Configuration, Cost]]]
]
