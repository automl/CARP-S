from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import json
from dataclasses import asdict
from pathlib import Path

from ConfigSpace import ConfigurationSpace
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from smacbenchmarking.benchmarks.problem import Problem


class AbstractLogger(Problem, ABC):
    def __init__(self, problem: Problem) -> None:
        self.problem: Problem = problem

    @abstractmethod
    def write_buffer(self) -> None:
        ...

    @abstractmethod
    def trial_to_buffer(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        ...

    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        trial_value = self.problem.evaluate(trial_info)
        self.trial_to_buffer(trial_info=trial_info, trial_value=trial_value)
        self.write_buffer()
        return trial_value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.problem, name)

    @property
    def configspace(self) -> ConfigurationSpace:
        return self.problem.configspace


class FileLogger(AbstractLogger):
    _filename: str = "trial_logs.jsonl"

    def __init__(self, problem: Problem, outdir: str) -> None:
        super().__init__(problem)

        self.outdir = outdir
        self.buffer: list[str] = []
        self.filename = Path(self.outdir) / self._filename

        print("Logging to ", self.filename)

    def write_buffer(self) -> None:
        if self.buffer:
            with open(self.filename, mode="a") as file:
                file.writelines(self.buffer)
            self.buffer = []

    def trial_to_buffer(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        info = {"trial_info": asdict(trial_info), "trial_value": asdict(trial_value)}
        info["trial_info"]["config"] = list(dict(info["trial_info"]["config"]).values())  # TODO beautify serialization
        info_str = json.dumps(info)
        self.buffer.append(info_str)
