from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import json
from dataclasses import asdict
from pathlib import Path
from omegaconf import DictConfig

from ConfigSpace import ConfigurationSpace
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
        self.n_trials: int = 0

    @abstractmethod
    def write_buffer(self) -> None:
        """Write buffer to destination.
        """
        ...

    @abstractmethod
    def trial_to_buffer(self, trial_info: TrialInfo, trial_value: TrialValue, n_trials: int | None = None) -> None:
        """Move the trial to the buffer.

        Parameters
        ----------
        trial_info : TrialInfo
            The trial info.
        trial_value : TrialValue
            The trial value.
        n_trials : int | None, optional
            The number of function evaluations, by default None. Starts with 1.
        """
        ...

    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate the problem and log the trial.

        Parameters
        ----------
        trial_info : TrialInfo
            Trial info.

        Returns
        -------
        TrialValue
            Trial value.
        """
        trial_value = self.problem.evaluate(trial_info)
        self.n_trials += 1
        self.trial_to_buffer(trial_info=trial_info, trial_value=trial_value, n_trials=self.n_trials)
        self.write_buffer()
        return trial_value

    def __getattr__(self, name: str) -> Any:
        """Get attribute of the wrapped problem.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        Any
            Attribute value.
        """
        return getattr(self.problem, name)

    @property
    def configspace(self) -> ConfigurationSpace:
        """Configuration Space

        All optimizers need to receive a configspace and
        convert it to their search space definition.

        Returns
        -------
        ConfigurationSpace
            Configuration space.
        """
        return self.problem.configspace


class FileLogger(AbstractLogger):
    _filename: str = "trial_logs.jsonl"

    def __init__(self, problem: Problem, cfg: DictConfig, outdir: str) -> None:
        """File logger.

        For each trial/function evaluate, write one line to the file.
        This line contains the json serialized info dict with `n_trials`,
        `trial_info` and `trial_value`.

        Parameters
        ----------
        problem : Problem
            The optimization problem.
        cfg : DictConfig
            Global experiment configuration.
        outdir : str
            Output directory.

        Attributes
        ----------
        problem : Problem
        cfg : DictConfig
        n_trials : int
            The number of function evaluations.
        outdir : str
            Output directory.
        buffer : list[str]
            Buffer of evaluation info. Json serialized info dict.
        _filename : str
            The name of the file, by default "trial_logs.jsonl".
        filename : str
            The path to the file with the proper output directory.
        """
        super().__init__(problem=problem, cfg=cfg)

        self.outdir = outdir
        self.buffer: list[str] = []
        self.filename = Path(self.outdir) / self._filename

    def write_buffer(self) -> None:
        """Write buffer to file.
        """
        if self.buffer:
            with open(self.filename, mode="a") as file:
                file.writelines(self.buffer)
            self.buffer = []

    def trial_to_buffer(self, trial_info: TrialInfo, trial_value: TrialValue, n_trials: int | None = None) -> None:
        """Evaluate the problem and log the trial.

        Parameters
        ----------
        trial_info : TrialInfo
            Trial info.

        Returns
        -------
        TrialValue
            Trial value.
        """
        info = {"n_trials": n_trials, "trial_info": asdict(trial_info), "trial_value": asdict(trial_value)}
        info["trial_info"]["config"] = list(dict(info["trial_info"]["config"]).values())  # TODO beautify serialization
        info_str = json.dumps(info) + "\n"
        self.buffer.append(info_str)
