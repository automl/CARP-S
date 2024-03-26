from __future__ import annotations

from abc import ABC, abstractmethod
from time import time
from typing import Any

from ConfigSpace import ConfigurationSpace

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.utils.trials import TrialInfo, TrialValue

SearchSpace = Any


class Optimizer(ABC):
    def __init__(self, problem: Problem, n_trials: int | None, time_budget: float | None) -> None:
        self.problem = problem
        if n_trials is None and time_budget is None:
            raise ValueError("Please specify either `n_trials` or `time_budget` "
                             "as the optimization budget.")
        self.n_trials: int | None = n_trials
        self.time_budget: float | None = time_budget
        
        super().__init__()
        # This indicates if the optimizer can deal with multi-fidelity optimization
        self.fidelity_enabled = False

        self._solver: Any = None

    @property
    def solver(self) -> Any:
        return self._solver
    
    @solver.setter
    def solver(self, value: Any) -> None:
        self._solver = value

    def setup_optimizer(self):
        self.solver = self._setup_optimizer()

    @abstractmethod
    def _setup_optimizer(self) -> Any:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def convert_to_trial(self, *args: tuple, **kwargs: dict) -> TrialInfo:
        """Convert proposal by optimizer to TrialInfo.

        This ensures that the problem can be evaluated with a unified API.

        Returns
        -------
        TrialInfo
            Trial info containing configuration, budget, seed, instance.
        """
        raise NotImplementedError
    
    def run(self) -> None:
        if self.solver is None:
            self.setup_optimizer()
        self._run()

    def _run(self) -> None:
        """Run Optimizer on Problem"""
        timeout = self.cfg.timeout

        start_time = time()
        while time() - start_time < timeout:
            trial = self.ask()
            result = self.problem.evaluate(trial)
            self.tell(result)
    
    @abstractmethod
    def ask(self) -> TrialInfo:
        """Ask the optimizer for a new trial to evaluate.

        If the optimizer does not support ask and tell,
        raise `smacbenchmarking.utils.exceptions.AskAndTellNotSupportedError`
        in child class.

        Returns
        -------
        TrialInfo
            trial info (config, seed, instance, budget)
        """
        raise NotImplementedError
    
    @abstractmethod
    def tell(self, trial_value: TrialValue) -> None:
        """Tell the optimizer a new trial.

        If the optimizer does not support ask and tell,
        raise `smacbenchmarking.utils.exceptions.AskAndTellNotSupportedError`
        in child class.

        Parameters
        ----------
        trial_value : TrialValue
            trial value (cost, time, ...)
        """
        raise NotImplementedError
