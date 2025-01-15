"""Calculate the Pareto front of a set of points."""

from __future__ import annotations

import numpy as np


def pareto(costs: np.ndarray) -> np.ndarray:
    """Find the Pareto front of a set of points.

    Parameters
    ----------
    costs : np.ndarray
        The costs of the points. Each row is a point, and each column is a
        dimension (N,D).

    Returns:
    --------
    np.ndarray
        A boolean array indicating which points are on the Pareto front (N,).
    """
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(costs[is_pareto] < c, axis=1)
            is_pareto[i] = True
    return is_pareto
