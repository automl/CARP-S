"""Pymoo Problem."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pymoo
import pymoo.problems
from ConfigSpace import ConfigurationSpace, Float
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.problems.multi.sympart import SYMPART, SYMPARTRotated

from carps.benchmarks.problem import Problem
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger

extra_probs = {
    "sympart": SYMPART,
    "sympart_rotated": SYMPARTRotated,
    "omnitest": OmniTest,
}


class PymooProblem(Problem):
    """Pymoo Problem class."""

    def __init__(
        self,
        problem_name: str,
        seed: int,
        metric: str | list[str],
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        """Initialize a Pymoo problem."""
        super().__init__(loggers)

        self.problem_name = problem_name
        self.metric = metric
        if problem_name in extra_probs:
            self._problem = extra_probs[problem_name]()
        else:
            self._problem = pymoo.problems.get_problem(self.problem_name)
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
        configspace.add_hyperparameters(hps)
        return configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate problem.

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
        costs = self._problem.evaluate(configuration).tolist()
        if len(costs) == 1:
            costs = costs[0]
        end_time = time.time()

        return TrialValue(
            cost=costs,
            time=end_time - start_time,
            starttime=start_time,
            endtime=end_time,
        )
