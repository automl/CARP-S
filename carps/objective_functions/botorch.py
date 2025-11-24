"""BOTorch Style Objective Functions."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
from ConfigSpace import Categorical, ConfigurationSpace, Constant, Float

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
        bounds = botorch_problem.bounds
        bound_len = bounds.shape[-1]
        fidelity_hp = None
        if hasattr(botorch_problem, "fidelities"):
            max_fidelity = botorch_problem.fidelities[-1]
            bounds = bounds[:, :-1]  # Last bound is for fidelity
            fidelity_hp = Constant(name="fidelity", value=max_fidelity)
        n_hps = len(botorch_problem.categorical_inds) + len(botorch_problem.continuous_inds)
        if fidelity_hp:
            n_hps -= 1
        hps = [1] * n_hps
        hp_order = [""] * n_hps
        for index in botorch_problem.continuous_inds:
            if index == bound_len - 1 and fidelity_hp:  # Account for potential fidelity
                continue
            lowerbound = float(bounds[0, index])
            upperbound = float(bounds[1, index])
            name = f"x{index}_cont"
            hp = Float(name, (lowerbound, upperbound))
            hps[index] = hp
            hp_order[index] = name
        for index in botorch_problem.categorical_inds:
            if index == bound_len - 1 and fidelity_hp:  # Account for potential fidelity
                continue
            lowerbound = int(bounds[0, index])
            upperbound = int(bounds[1, index])
            items = list(range(lowerbound, upperbound + 1))
            name = f"x{index}_cat"
            hp = Categorical(name, items=items)
            hps[index] = hp
            hp_order[index] = name
        if fidelity_hp is not None:
            hps.append(fidelity_hp)
            hp_order.append("fidelity")
        self.hp_order = hp_order
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
        config_dict = dict(configuration)
        config_tensor = torch.tensor([[config_dict[k] for k in self.hp_order]])

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
