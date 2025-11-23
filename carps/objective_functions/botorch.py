"""BOTorch Style Objective Functions."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
from ConfigSpace import Categorical, ConfigurationSpace, Float

from carps.objective_functions.objective_function import ObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from botorch.test_functions.base import BaseTestProblem

    from carps.loggers.abstract_logger import AbstractLogger


class BOTorchObjectiveFunction(ObjectiveFunction):
    """HPOBench objective function."""

    def __init__(
        self,
        seed: int,
        botorch_problem: BaseTestProblem,
        loggers: list[AbstractLogger] | None = None,
    ):
        """Initialize BOTorch style objective function.

        Parameters
        ----------
        seed : int
            The seed, currently not passed.
        botorch_problem : BaseTestProblem
            The botorch problem.
        loggers : list[AbstractLogger] | None, optional
            Loggers, by default None
        """
        super().__init__(loggers)

        self.botorch_problem = botorch_problem

        # Create configspace
        assert len(botorch_problem.discrete_inds) == 0
        hps = []
        for index in botorch_problem.continuous_inds:
            lowerbound = float(botorch_problem.bounds[0, index])
            upperbound = float(botorch_problem.bounds[1, index])
            hp = Float(f"x{index}_cont", (lowerbound, upperbound))
            hps.append(hp)
        for index in botorch_problem.categorical_inds:
            lowerbound = int(botorch_problem.bounds[0, index])
            upperbound = int(botorch_problem.bounds[1, index])
            items = list(range(lowerbound, upperbound + 1))
            hp = Categorical(f"x{index}_cat", items=items)
            hps.append(hp)
        self._configspace = ConfigurationSpace(space=hps)
        self.seed = seed  # unused

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
        configuration = trial_info.config
        starttime = time.time()

        config_tensor = torch.tensor(list(configuration.values()))

        cost = float(self.botorch_problem.evaluate_true(config_tensor.reshape(1, -1)))

        endtime = time.time()
        T = endtime - starttime
        virtual_time = 0.0
        # function_value is 1 - accuracy on the validation set
        return TrialValue(
            cost=cost,
            time=T,
            starttime=starttime,
            endtime=endtime,
            virtual_time=virtual_time,
        )
