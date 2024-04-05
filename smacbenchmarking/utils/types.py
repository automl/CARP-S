from __future__ import annotations

from typing import Any, TypeAlias 
import numpy as np
from ConfigSpace import Configuration

SearchSpace: TypeAlias  = Any
Cost: TypeAlias  = np.ndarray | float
Incumbent: TypeAlias  = tuple[Configuration, Cost] | list[tuple[Configuration, Cost]] | None