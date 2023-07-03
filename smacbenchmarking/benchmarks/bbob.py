from typing import Any

import ioh
from ConfigSpace import ConfigurationSpace, Float
from smac.runhistory.dataclasses import TrialInfo

from smacbenchmarking.benchmarks.problem import SingleObjectiveProblem


class BBOBProblem(SingleObjectiveProblem):
    def __init__(self, fid: int, instance: int, dimension: int, seed: int):
        super().__init__()

        self._configspace, self._problem = get_bbob_problem(fid=fid, instance=instance, dimension=dimension, seed=seed)

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._configspace

    def evaluate(self, trial_info: TrialInfo) -> float:
        configuration = trial_info.config
        input = list(dict(configuration).values())
        output = self._problem(input)
        return output


def get_bbob_problem(fid: int, instance: int, dimension: int, seed: int) -> tuple[ConfigurationSpace, Any]:
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
