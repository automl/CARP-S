from __future__ import annotations

from abc import ABC, abstractmethod
from omegaconf import DictConfig
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from smacbenchmarking.benchmarks.problem import Problem

# TODO log optimizer's execution time

class AbstractLogger(Problem, ABC):
    def __init__(self, problem: Problem, cfg: DictConfig) -> None:
        """AbstractLogger

        Wraps `Problem` and intercepts the trial info and value
        during evaluate.

        Parameters
        ----------
        problem : Problem
            The optimization problem.
        cfg : DictConfig
            The global experiment configuration. Might be relevant for
            some loggers.

        Attributes
        ----------
        problem : Problem
        cfg : DictConfig
        n_trials : int
            The number of function evaluations.
        """
        self.problem: Problem = problem
        self.cfg = cfg

    @abstractmethod
    def log_trial(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Log the trial.

        Parameters
        ----------
        trial_info : TrialInfo
            The trial info.
        trial_value : TrialValue
            The trial value.
        """
        raise NotImplementedError
