from __future__ import annotations
from typing import Any

from ConfigSpace import ConfigurationSpace
from smacbenchmarking.optimizers.optimizer import Optimizer
from smacbenchmarking.utils.trials import TrialInfo, TrialValue

SearchSpace = Any

class OptimizerWrapper(Optimizer):
    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def __getattribute__(self, name: str) -> Any:
        """
        Get attribute value of wrapper if available and of optimizer if not.

        Parameters
        ----------
        name : str
            Attribute to get

        Returns
        -------
        value
            Value of given name

        """
        if name in ["optimizer"]:
            return object.__getattribute__(self, name)
        else:
            return getattr(self.env, name)

    @property
    def unwrapped(self) -> Optimizer:
        """Returns the base optimizer of the wrapper.

        This will be the bare :class:`smacbenchmarking.optimizers.optimizer.Optimizer`, underneath all layers of wrappers.
        """
        return self.optimizer.unwrapped
    
    def ask(self) -> TrialInfo:
        # TODO Copy docstring
        return self.optimizer.ask()
    
    def tell(self, trial_value: TrialValue) -> None:
        # TODO Copy docstring
        self.optimizer.tell(trial_value=trial_value)

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        """Convert ConfigSpace configuration space to search space from optimizer.

        Parameters
        ----------
        configspace : ConfigurationSpace
            Configuration space from Problem.

        Returns
        -------
        SearchSpace
            Optimizer's search space.
        """
        return self.optimizer.convert_configspace(configspace)
    
    def convert_to_trial(self, *args: tuple, **kwargs: dict) -> TrialInfo:
        """Convert proposal by optimizer to TrialInfo.

        This ensures that the problem can be evaluated with a unified API.

        Returns
        -------
        TrialInfo
            Trial info containing configuration, budget, seed, instance.
        """
        return self.optimizer.convert_to_trial(*args, **kwargs)
    
    def run(self) -> None:
        """Run Optimizer on Problem"""
        return self.optimizer.run()