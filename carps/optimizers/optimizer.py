"""Base class for all optimizers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from carps.utils.task import Task
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from carps.benchmarks.problem import Problem
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.types import Incumbent, SearchSpace


class Optimizer(ABC):
    """Base class for all optimizers."""

    def __init__(
        self, problem: Problem, task: Task | dict | DictConfig, loggers: list[AbstractLogger] | None = None
    ) -> None:
        """Optimizer.

        Parameters
        ----------
        problem : Problem
            Optimization problem aka the function to be optimized.
        task : Task | dict | DictConfig
            Task definition, e.g. specifiying the number of trials, etc.
        loggers : list[AbstractLogger] | None, optional
            Loggers, by default None

        Raises:
        ------
        ValueError
            Unknown task type, must be either `Task`, `dict` or `DictConfig`.
        """
        super().__init__()
        self.problem = problem

        if isinstance(task, dict):
            task = Task(**task)
        elif isinstance(task, DictConfig):
            task = Task(**OmegaConf.to_container(cfg=task, resolve=True))
        elif isinstance(task, Task):
            pass
        else:
            raise ValueError("task must be either `Task`, `dict` or `DictConfig`.")

        self.task: Task = task
        self.loggers: list[AbstractLogger] = loggers if loggers is not None else []

        # Convert min to seconds
        self.time_budget = self.task.time_budget * 60 if self.task.time_budget is not None else None
        self.virtual_time_elapsed_seconds: float = 0.0
        self.trial_counter: int | float = 0

        # This indicates if the optimizer can deal with multi-fidelity optimization
        self.fidelity_enabled = False

        self._solver: Any = None
        self._last_incumbent: tuple[TrialInfo, TrialValue] | None = None

    @property
    def solver(self) -> Any:
        """Solver instance.

        Returns.
        -------
        Any
            Solver instance.
        """
        return self._solver

    @solver.setter
    def solver(self, value: Any) -> None:
        """Set the solver instance.

        Parameters
        ----------
        value : Any
            Solver instance.
        """
        self._solver = value

    def setup_optimizer(self) -> None:
        """Setup the optimizer."""
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

        Returns:
        -------
        SearchSpace
            Optimizer's search space.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_to_trial(self, *args: tuple, **kwargs: dict) -> TrialInfo:
        """Convert proposal by optimizer to TrialInfo.

        This ensures that the problem can be evaluated with a unified API.

        Returns:
        -------
        TrialInfo
            Trial info containing configuration, budget, seed, instance.
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_incumbent(self) -> Incumbent:
        """Extract the incumbent config and cost. May only be available after a complete run.

        Returns:
        -------
        Incumbent: tuple[TrialInfo, TrialValue] | list[tuple[TrialInfo, TrialValue]] | None
            The incumbent configuration with associated cost.
        """
        raise NotImplementedError

    def run(self) -> Incumbent:
        """Run the optimizer.

        Setup if not already done and run the optimizer.

        Returns.
        -------
        Incumbent
            Best performing configuration(s).
        """
        if self.solver is None:
            self.setup_optimizer()
        return self._run()

    def _time_left(self, start_time: float) -> bool:
        if self.time_budget is not None:
            return (time.time() - start_time) + self.virtual_time_elapsed_seconds < self.time_budget
        return True

    def continue_optimization(self, start_time: float) -> bool:
        """Check whether to continue optimization.

        (a) Based on time budget if specified.
        (b) Based on number of trials if specified.

        Parameters
        ----------
        start_time : float
            Starting time.

        Returns:
        -------
        bool
            True when optimization should continue, false otherwise.
        """
        cont = True
        if self.time_budget is not None and not self._time_left(start_time):
            cont = False
        if self.task.n_trials is not None and self.trial_counter >= self.task.n_trials:
            cont = False

        return cont

    def _run(self) -> Incumbent:
        """Run Optimizer on Problem."""
        start_time = time.time()
        while self.continue_optimization(start_time=start_time):
            trial_info = self.ask()
            normalized_budget = 1.0
            if self.task.max_budget is not None and trial_info.budget is not None:
                normalized_budget = trial_info.budget / self.task.max_budget
            if self.task.is_multifidelity:
                trial_info = TrialInfo(
                    config=trial_info.config,
                    instance=trial_info.instance,
                    seed=trial_info.seed,
                    budget=trial_info.budget,
                    normalized_budget=normalized_budget,
                    checkpoint=trial_info.checkpoint,
                    name=trial_info.name,
                )
            trial_value = self.problem.evaluate(trial_info=trial_info)
            self.virtual_time_elapsed_seconds += trial_value.virtual_time
            self.tell(trial_info=trial_info, trial_value=trial_value)

            new_incumbent = self.get_current_incumbent()
            if new_incumbent != self._last_incumbent:
                self._last_incumbent = new_incumbent  # type: ignore[assignment]
                for logger in self.loggers:
                    logger.log_incumbent(self.trial_counter, new_incumbent)

            if not self.task.is_multifidelity:
                self.trial_counter += 1
            else:
                assert (
                    self.task.max_budget is not None
                ), "Define max_budget for multi-fidelity optimization in your problem setup."
                assert trial_info.budget is not None
                self.trial_counter += trial_info.budget / self.task.max_budget
        return self.get_current_incumbent()

    @abstractmethod
    def ask(self) -> TrialInfo:
        """Ask the optimizer for a new trial to evaluate.

        If the optimizer does not support ask and tell,
        raise `carps.utils.exceptions.AskAndTellNotSupportedError`
        in child class.

        Returns:
        -------
        TrialInfo
            trial info (config, seed, instance, budget)
        """
        raise NotImplementedError

    @abstractmethod
    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Tell the optimizer a new trial.

        If the optimizer does not support ask and tell,
        raise `carps.utils.exceptions.AskAndTellNotSupportedError`
        in child class.

        Parameters
        ----------
        trial_info : TrialInfo
            trial info (config, seed, instance, budget)
        trial_value : TrialValue
            trial value (cost, time, ...)
        """
        raise NotImplementedError
