"""Pymoo ObjectiveFunction."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pymoo
import pymoo.problems
from ConfigSpace import ConfigurationSpace, Float
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.problems.multi.sympart import SYMPART, SYMPARTRotated

from carps.objective_functions.objective_function import ObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger

extra_probs = {
    "sympart": SYMPART,
    "sympart_rotated": SYMPARTRotated,
    "omnitest": OmniTest,
}


class PymooObjectiveFunction(ObjectiveFunction):
    """Pymoo ObjectiveFunction class."""

    def __init__(
        self,
        problem_name: str,
        seed: int,
        problem_kwargs: dict | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        """Initialize a Pymoo problem."""
        if problem_kwargs is None:
            problem_kwargs = {}
        super().__init__(loggers)

        self.problem_name = problem_name
        if problem_name in extra_probs:
            self._problem = extra_probs[problem_name](**problem_kwargs)
        else:
            self._problem = pymoo.problems.get_problem(self.problem_name, **problem_kwargs)
        self._configspace = self.get_pymoo_space(pymoo_prob=self._problem, seed=seed)

    @property
    def configspace(self) -> ConfigurationSpace:
        """Return configuration space.

        Returns:
        -------
        ConfigurationSpace
            Configuration space.
        """
        return self._configspace

    def get_pymoo_space(self, pymoo_prob: pymoo.core.problem.Problem, seed: int) -> ConfigurationSpace:
        """Get ConfigSpace from pymoo problem."""
        n_var = pymoo_prob.n_var
        xl, xu = pymoo_prob.xl, pymoo_prob.xu
        hps = [Float(name=f"x{i}", bounds=[xl[i], xu[i]]) for i in range(n_var)]
        configspace = ConfigurationSpace(seed=seed)
        configspace.add(hps)
        return configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate objective function.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns:
        -------
        TrialValue
            Cost
        """
        configuration = np.array(list(trial_info.config.values()))
        start_time = time.time()
        # Ask for a dictionary rather than the positional return. `evaluate` hands back a bare array only for an
        # unconstrained problem; as soon as the problem has constraints it returns a tuple, which has no `tolist`.
        out = self._problem.evaluate(configuration, return_as_dictionary=True)
        end_time = time.time()

        costs = np.atleast_1d(out["F"]).tolist()
        if len(costs) == 1:
            costs = costs[0]

        return TrialValue(
            cost=costs,
            time=end_time - start_time,
            starttime=start_time,
            endtime=end_time,
            additional_info=self._constraint_values(out),
        )

    @staticmethod
    def _constraint_values(out: dict[str, Any]) -> dict[str, float]:
        """Names the constraint values pymoo reported, so that a bound can refer to one of them.

        Inequality constraints are `g0`, `g1`, ... and are satisfied when the value is at most zero; equality
        constraints are `h0`, `h1`, ... and are satisfied at zero. That is pymoo's own convention, kept as is
        rather than translated, so the numbers mean what the problem definition says they mean.

        Named scalars rather than the raw arrays: a bound has to be able to refer to one constraint, and an array
        would additionally need custom json encoding to survive the loggers.
        """
        values = {}

        for key, prefix in (("G", "g"), ("H", "h")):
            if key not in out or out[key] is None:
                continue

            for index, value in enumerate(np.atleast_1d(out[key])):
                values[f"{prefix}{index}"] = float(value)

        return values
