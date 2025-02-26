"""ManyAffineBBOB benchmark functions."""

from __future__ import annotations

from collections.abc import Sequence

# https://github.com/Dvermetten/Many-affine-BBOB/blob/master/affine_barebones.py
import ioh  # type: ignore
import numpy as np

scale_factors = [
    11.0,
    17.5,
    12.3,
    12.6,
    11.5,
    15.3,
    12.1,
    15.3,
    15.2,
    17.4,
    13.4,
    20.4,
    12.9,
    10.4,
    12.3,
    10.3,
    9.8,
    10.6,
    10.0,
    14.7,
    10.7,
    10.8,
    9.0,
    12.1,
]


class ManyAffine:
    """ManyAffineBBOB benchmark function."""

    def __init__(
        self,
        weights: np.ndarray,
        instances: Sequence[int],
        opt_loc: float | int = 1,
        dim: int = 5,
        sf_type: str = "min_max",  # noqa: ARG002
    ) -> None:
        """Initialize ManyAffineBBOB benchmark function.

        Parameters
        ----------
        weights : np.ndarray
            Weights for the functions.
        instances : Sequence[int]
            Instances for the functions.
        opt_loc : int | float, optional
            Optimum location, by default 1.
        dim : int, optional
            Dimension of the function, by default 5.
        sf_type : str, optional
            Scale factor type, by default "min_max".
        """
        self.weights = weights / np.sum(weights)
        self.fcts = [ioh.get_problem(fid, int(iid), dim) for fid, iid in zip(range(1, 25), instances, strict=False)]
        self.opts = [f.optimum.y for f in self.fcts]
        self.scale_factors = scale_factors
        if isinstance(opt_loc, int):
            self.opt_x = self.fcts[opt_loc].optimum.x
        else:
            self.opt_x = opt_loc

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate ManyAffineBBOB benchmark function.

        Parameters
        ----------
        x : np.ndarray
            Input.

        Returns:
        -------
        float
            Output.
        """
        raw_vals = np.array(
            [
                np.clip(f(x + f.optimum.x - self.opt_x) - o, 1e-12, 1e20)
                for f, o in zip(self.fcts, self.opts, strict=False)
            ]
        )
        weighted = (np.log10(raw_vals) + 8) / self.scale_factors * self.weights
        return 10 ** (10 * np.sum(weighted) - 8)


def register_many_affine_functions():
    """Register ManyAffineBBOB benchmark functions in ioh."""
    n_functions = 24
    dimensions = np.arange(2, 11)

    seed = 476926
    rng = np.random.default_rng(seed=seed)

    for dim in dimensions:
        for i in range(1, n_functions + 1):  # 1-based, same as original bbob
            weights = rng.uniform(size=24)
            iids = rng.integers(100, size=24)
            opt_loc = rng.uniform(size=dim) * 10 - 5  # in [-5,5] as BBOB

            f_new = ManyAffine(weights, iids, opt_loc, dim)

            name = f"MA{i}_d{dim}"

            ioh.problem.wrap_real_problem(f_new, name=name, optimization_type=ioh.OptimizationType.MIN, lb=-5, ub=5)
