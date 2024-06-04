from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import ioh
from ConfigSpace import ConfigurationSpace, Float

from carps.benchmarks.manyaffinebbob import register_many_affine_functions
from carps.benchmarks.problem import Problem
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger

register_many_affine_functions()


class BBOBProblem(Problem):
    def __init__(self, fid: int, instance: int, dimension: int, seed: int, loggers: list[AbstractLogger] | None = None):
        super().__init__(loggers)

        self._configspace, self._problem = get_bbob_problem(fid=fid, instance=instance, dimension=dimension, seed=seed)

    @property
    def f_min(self) -> float | None:
        return self._problem.optimum.y

    @property
    def configspace(self) -> ConfigurationSpace:
        """Return configuration space.

        Returns:
        -------
        ConfigurationSpace
            Configuration space.
        """
        return self._configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate problem.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns:
        -------
        float
            Cost
        """
        configuration = trial_info.config
        input = list(dict(configuration).values())
        starttime = time.time()
        cost = self._problem(input)
        endtime = time.time()
        T = endtime - starttime

        return TrialValue(cost=cost, time=T, starttime=starttime, endtime=endtime)


def get_bbob_problem(fid: int, instance: int, dimension: int, seed: int) -> tuple[ConfigurationSpace, Any]:
    r"""Get BBOB problem.

    Parameters
    ----------
    fid : int
        Function id $\in$ [1,24].
    instance : int
        Function instance.
    dimension : int
        Dimension. 1-x
    seed : int
        Seed for configuration space.

    Returns:
    -------
    tuple[ConfigurationSpace, Any]
        Configuration space, target function.
    """
    problem = ioh.get_problem(
        fid=fid,
        instance=instance,
        dimension=dimension,
        # problem_type=ProblemType.BBOB,
    )

    # Configuration space
    lower_bounds = problem.bounds.lb
    upper_bounds = problem.bounds.ub
    n_dim = problem.meta_data.n_variables
    hps = [Float(name=f"x{i}", bounds=[lower_bounds[i], upper_bounds[i]]) for i in range(n_dim)]
    configuration_space = ConfigurationSpace(seed=seed)
    configuration_space.add_hyperparameters(hps)

    return configuration_space, problem
