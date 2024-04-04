from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any
import numpy as np

from ConfigSpace import ConfigurationSpace, Configuration

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.utils.trials import TrialInfo, TrialValue

SearchSpace = Any
Cost = np.ndarray | float
Incumbent = tuple[Configuration, Cost] | list[tuple[Configuration, Cost]] | None


class Optimizer(ABC):
    def __init__(self, problem: Problem, n_trials: int | None, time_budget: float | None = None) -> None:
        self.problem = problem
        if n_trials is None and time_budget is None:
            raise ValueError("Please specify either `n_trials` or `time_budget` "
                             "as the optimization budget.")
        self.n_trials: int | None = n_trials
        self.time_budget: float | None = time_budget
        self.virtual_time_elapsed_seconds: float | None = 0.0
        self.trial_counter: int = 0

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

    @abstractmethod
    def extract_incumbent(self) -> Incumbent:
        """Extract the incumbent config and cost after run.

        Returns
        -------
        Incumbent: tuple[TrialInfo, TrialValue] | list[tuple[TrialInfo, TrialValue]] | None
            The incumbent configuration with associated cost.
        """
        raise NotImplementedError

    def run(self) -> Incumbent:
        if self.solver is None:
            self.setup_optimizer()
        return self._run()

    def _time_left(self, start_time) -> float:
        return (time.time() - start_time) + self.virtual_time_elapsed_seconds < self.time_budget

    def continue_optimization(self, start_time) -> bool:
        cont = True
        if self.time_budget is not None and not self._time_left(start_time):
            cont = False
        if self.trial_counter >= self.n_trials:
            cont = False

        return cont

    def _run(self) -> Incumbent:
        """Run Optimizer on Problem"""
        start_time = time.time()
        while self.continue_optimization(start_time=start_time):
            trial_info = self.ask()
            trial_value = self.problem.evaluate(trial_info=trial_info)
            self.virtual_time_elapsed_seconds += trial_value.virtual_time
            self.tell(trial_info=trial_info, trial_value=trial_value)
            self.trial_counter += 1

        return self.extract_incumbent()

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
    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Tell the optimizer a new trial.

        If the optimizer does not support ask and tell,
        raise `smacbenchmarking.utils.exceptions.AskAndTellNotSupportedError`
        in child class.

        Parameters
        ----------
        trial_info : TrialInfo
            trial info (config, seed, instance, budget)
        trial_value : TrialValue
            trial value (cost, time, ...)
        """
        raise NotImplementedError
