"""Synthetic objective functions with unknown constraints.

The benchmarks the constrained Bayesian optimization literature is written against. They are small and
analytic, and each one is deliberately shaped so that the unconstrained optimum is unhelpful: the
constraint either removes it outright or sits right against it, so a method that ignores feasibility
looks good for a while and then has nothing to show.

pymoo's G-suite covers constrained evolutionary optimization; these are the ones the constrained BO
papers use, and none of them is in pymoo.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from ConfigSpace import ConfigurationSpace, Float

from carps.objective_functions.objective_function import ObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger

__all__ = ["PROBLEMS", "ConstrainedSyntheticObjectiveFunction", "ConstrainedSyntheticProblem"]


@dataclass(frozen=True)
class ConstrainedSyntheticProblem:
    """A synthetic objective with named constraints on quantities the objective does not include.

    Parameters
    ----------
    name : str
        Problem name.
    bounds : list[tuple[float, float]]
        Lower and upper bound per input dimension.
    objective : Callable[[np.ndarray], float]
        The quantity to minimize.
    constraints : dict[str, Callable[[np.ndarray], float]]
        The measured quantities the constraints are stated over, by name.
    bounds_on_constraints : dict[str, str]
        The bound each constrained quantity has to respect, as `"<= 50.0"` or `">= 0.5"`.
    reference : str
        Where the problem is from.
    """

    name: str
    bounds: list[tuple[float, float]]
    objective: Callable[[np.ndarray], float]
    constraints: dict[str, Callable[[np.ndarray], float]] = field(default_factory=dict)
    bounds_on_constraints: dict[str, str] = field(default_factory=dict)
    reference: str = ""

    @property
    def dimensions(self) -> int:
        """Number of input dimensions."""
        return len(self.bounds)

    @property
    def constraint_expressions(self) -> list[str]:
        """The constraints as SMAC-style expressions, e.g. `"disk <= 50.0"`."""
        return [f"{name} {self.bounds_on_constraints[name]}" for name in self.constraints]

    def configspace(self, seed: int = 0) -> ConfigurationSpace:
        """Configuration space of the problem."""
        configspace = ConfigurationSpace(seed=seed)
        configspace.add([Float(name=f"x{i}", bounds=list(b)) for i, b in enumerate(self.bounds)])

        return configspace

    def is_feasible(self, x: np.ndarray) -> bool:
        """Whether a point respects every bound."""
        for name, function in self.constraints.items():
            operator, bound = self.bounds_on_constraints[name].split()
            value = function(x)
            if operator == "<=" and not value <= float(bound):
                return False
            if operator == ">=" and not value >= float(bound):
                return False

        return True

    def _feasible_grid(self, resolution: int) -> np.ndarray:
        """The feasible points of a dense grid over the search space."""
        axes = [np.linspace(lower, upper, resolution) for lower, upper in self.bounds]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.stack([m.ravel() for m in mesh], axis=1)

        feasible = np.ones(len(points), dtype=bool)
        for name, function in self.constraints.items():
            operator, bound = self.bounds_on_constraints[name].split()
            values = function(points.T)
            feasible &= values <= float(bound) if operator == "<=" else values >= float(bound)

        if not feasible.any():
            raise ValueError(f"No feasible point found for {self.name} at resolution {resolution}.")

        return points[feasible]

    def constrained_optimum(self, resolution: int = 2000) -> float:
        """The best feasible objective value, found on a dense grid.

        Computed rather than recorded, so that the number a task config carries always matches the
        functions defined here. `resolution` is per dimension; these problems are two-dimensional.
        """
        points = self._feasible_grid(resolution)

        return float(np.min(self.objective(points.T)))

    def constrained_argmin(self, resolution: int = 2000) -> dict[str, float]:
        """Where the best feasible value is attained, by hyperparameter name.

        A benchmark that states where its optimum lies can be used to build a user prior of a known
        quality, which is how the prior-guided literature evaluates its methods.
        """
        points = self._feasible_grid(resolution)
        best = points[int(np.argmin(self.objective(points.T)))]

        return {f"x{i}": float(value) for i, value in enumerate(best)}


def _branin(x: np.ndarray) -> np.ndarray:
    a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi, 6.0, 10.0, 1 / (8 * np.pi)

    return a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s


PROBLEMS: dict[str, ConstrainedSyntheticProblem] = {
    # The disk removes two of Branin's three global minima, leaving the one at (pi, 2.275). A method that
    # ignores feasibility converges happily to a minimum it is not allowed to use.
    "branin_disk": ConstrainedSyntheticProblem(
        name="branin_disk",
        bounds=[(-5.0, 10.0), (0.0, 15.0)],
        objective=_branin,
        constraints={"disk": lambda x: (x[0] - 2.5) ** 2 + (x[1] - 7.5) ** 2},
        bounds_on_constraints={"disk": "<= 50.0"},
        reference="Gelbart, Snoek & Adams, Bayesian Optimization with Unknown Constraints, UAI 2014",
    ),
    "gardner1": ConstrainedSyntheticProblem(
        name="gardner1",
        bounds=[(0.0, 6.0), (0.0, 6.0)],
        objective=lambda x: np.cos(2 * x[0]) * np.cos(x[1]) + np.sin(x[0]),
        # cos(x0)cos(x1) - sin(x0)sin(x1) is cos(x0 + x1); the bound is what makes the constraint binding.
        # With it, the best feasible value is -1.8887, against -2 unconstrained - the published pair, up to
        # the sign convention of stating the problem as a minimization.
        constraints={"c": lambda x: np.cos(x[0]) * np.cos(x[1]) - np.sin(x[0]) * np.sin(x[1])},
        bounds_on_constraints={"c": "<= -0.5"},
        reference="Gardner et al., Bayesian Optimization with Inequality Constraints, ICML 2014",
    ),
    "gardner2": ConstrainedSyntheticProblem(
        name="gardner2",
        bounds=[(0.0, 6.0), (0.0, 6.0)],
        objective=lambda x: np.sin(x[0]) + x[1],
        constraints={"c": lambda x: np.sin(x[0]) * np.sin(x[1])},
        bounds_on_constraints={"c": "<= -0.95"},
        reference="Gardner et al., Bayesian Optimization with Inequality Constraints, ICML 2014",
    ),
    # Two constraints, one of them oscillating, so the feasible region is disconnected near the optimum.
    "gramacy": ConstrainedSyntheticProblem(
        name="gramacy",
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        objective=lambda x: x[0] + x[1],
        constraints={
            "c1": lambda x: 1.5 - x[0] - 2 * x[1] - 0.5 * np.sin(2 * np.pi * (x[0] ** 2 - 2 * x[1])),
            "c2": lambda x: x[0] ** 2 + x[1] ** 2 - 1.5,
        },
        bounds_on_constraints={"c1": "<= 0.0", "c2": "<= 0.0"},
        reference="Gramacy et al., Modeling an augmented Lagrangian for blackbox constrained optimization, 2016",
    ),
}


class ConstrainedSyntheticObjectiveFunction(ObjectiveFunction):
    """A synthetic objective function with unknown constraints.

    The constrained quantities are reported alongside the cost, under the names the task's bounds refer
    to, exactly as a real target function would report a measured latency or model size.

    Parameters
    ----------
    problem_name : str
        One of the keys of `PROBLEMS`.
    seed : int
        Seed for the configuration space.
    loggers : list[AbstractLogger] | None, defaults to None
        Loggers.
    """

    def __init__(self, problem_name: str, seed: int, loggers: list[AbstractLogger] | None = None) -> None:
        """Initialize a constrained synthetic objective function."""
        super().__init__(loggers)

        if problem_name not in PROBLEMS:
            raise ValueError(f"Unknown problem {problem_name!r}. Available problems are {sorted(PROBLEMS)}.")

        self.problem_name = problem_name
        self._problem = PROBLEMS[problem_name]
        self._configspace = self._problem.configspace(seed=seed)

    @property
    def configspace(self) -> ConfigurationSpace:
        """Configuration space of the problem."""
        return self._configspace

    @property
    def f_min(self) -> float | None:
        """The best feasible objective value."""
        return self._problem.constrained_optimum()

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate the objective and the constrained quantities.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns:
        -------
        TrialValue
            The cost, with the constrained quantities in `additional_info`.
        """
        x = np.array(list(trial_info.config.values()))

        start_time = time.time()
        cost = float(self._problem.objective(x))
        constraint_values = {name: float(function(x)) for name, function in self._problem.constraints.items()}
        end_time = time.time()

        return TrialValue(
            cost=cost,
            time=end_time - start_time,
            starttime=start_time,
            endtime=end_time,
            additional_info=constraint_values,
        )
