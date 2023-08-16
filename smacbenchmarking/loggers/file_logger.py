from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from omegaconf import DictConfig
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.loggers.abstract_logger import AbstractLogger


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
        """Write buffer to file."""
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
