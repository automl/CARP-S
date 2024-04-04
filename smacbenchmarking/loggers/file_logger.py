from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode

from smacbenchmarking.loggers.abstract_logger import AbstractLogger
from smacbenchmarking.optimizers.optimizer import Incumbent
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


def dump_logs(log_data: dict, filename: str):
    """Dump log dict in jsonl format

    This appends one json dict line to the filename.

    Parameters
    ----------
    log_data : dict
        Data to log, must be json serializable.
    filename : str
        Filename without path. The path will be either the
        current working directory or if it is called during
        a hydra session, the hydra run dir will be the log
        dir.
    """
    log_data_str = json.dumps(log_data) + "\n"

    try:
        # Check if we are in a hydra context
        hydra_cfg = HydraConfig.instance().get()
        if hydra_cfg.mode == RunMode.RUN:
            directory = Path(hydra_cfg.run.dir)
        else:  # MULTIRUN
            directory = Path(hydra_cfg.sweep.dir) / hydra_cfg.sweep.subdir
    except Exception:
        directory = "."
    filename = Path(directory) / filename
    with open(filename, mode="a") as file:
        file.writelines([log_data_str])


class FileLogger(AbstractLogger):
    def __init__(self) -> None:
        """File logger.

        For each trial/function evaluate, write one line to the file.
        This line contains the json serialized info dict with `n_trials`,
        `trial_info` and `trial_value`.
        """
        super().__init__()

    def log_trial(self, n_trials: int, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Evaluate the problem and log the trial.

        Parameters
        ----------
        n_trials : int
            Number of trials that have been run so far.
        trial_info : TrialInfo
            Trial info.
        trial_value : TrialValue
            Trial value.
        """
        info = {"n_trials": n_trials, "trial_info": asdict(trial_info), "trial_value": asdict(trial_value)}
        info["trial_info"]["config"] = list(dict(info["trial_info"]["config"]).values())
        info_str = json.dumps(info) + "\n"
        logging.info(info_str)

        dump_logs(log_data=info, filename="trial_logs.jsonl")

    def log_incumbent(self, incumbent: Incumbent) -> None:
        pass

    def log_arbitrary(self, data: dict, entity: str) -> None:
        pass
