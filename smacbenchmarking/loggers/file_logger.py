from __future__ import annotations

import json
from dataclasses import asdict

from omegaconf import DictConfig

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.loggers.abstract_logger import AbstractLogger

import logging

from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class FileLogger(AbstractLogger):

    def __init__(self, problem: Problem, cfg: DictConfig) -> None:
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
        """
        super().__init__(problem=problem, cfg=cfg)


    def log_trial(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Evaluate the problem and log the trial.

        Parameters
        ----------
        trial_info : TrialInfo
            Trial info.
        trial_value : TrialValue
            Trial value.
        """
        info = {"trial_info": asdict(trial_info), "trial_value": asdict(trial_value)}
        info["trial_info"]["config"] = list(dict(info["trial_info"]["config"]).values())  # TODO beautify serialization
        info_str = json.dumps(info) + "\n"
        logging.info(info_str)
