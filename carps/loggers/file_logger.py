from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode


from carps.loggers.abstract_logger import AbstractLogger
from carps.optimizers.optimizer import Incumbent
from carps.utils.trials import TrialInfo, TrialValue
from carps.utils.logging import setup_logging
from carps.utils.logging import get_logger

setup_logging()
logger = get_logger("FileLogger")


def get_run_directory() -> str:
    """Get current run dir

    Either '.' if hydra not active, else hydra run dir.

    Returns
    -------
    str
        Directory
    """
    try:
        # Check if we are in a hydra context
        hydra_cfg = HydraConfig.instance().get()
        if hydra_cfg.mode == RunMode.RUN:
            directory = Path(hydra_cfg.run.dir)
        else:  # MULTIRUN
            directory = Path(hydra_cfg.sweep.dir) / hydra_cfg.sweep.subdir
    except Exception:
        directory = "."
    return directory


def dump_logs(log_data: dict, filename: str, directory: str | None = None):
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
    directory: str | None, defaults to None
        Directory to log to. If None, either use hydra run dir or current dir.
    """
    log_data_str = json.dumps(log_data) + "\n"
    directory = directory or get_run_directory()
    filename = Path(directory) / filename
    with open(filename, mode="a") as file:
        file.writelines([log_data_str])


def convert_trials(n_trials, trial_info, trial_value):
    info = {"n_trials": n_trials, "trial_info": asdict(trial_info), "trial_value": asdict(trial_value)}
    info["trial_info"]["config"] = list(dict(info["trial_info"]["config"]).values())
    return info


class FileLogger(AbstractLogger):
    _filename: str = "trial_logs.jsonl"
    _filename_trajectory: str = "trajectory_logs.jsonl"

    def __init__(self, overwrite: bool = False, directory: str | None = None) -> None:
        """File logger.

        For each trial/function evaluate, write one line to the file.
        This line contains the json serialized info dict with `n_trials`,
        `trial_info` and `trial_value`.

        Parameters
        ------
        overwrite: bool, defaults to True
            Delete previous logs in that directory if True.
            If false, raise an error message.
        directory: str | None, defaults to None
            Directory to log to. If None, either use hydra run dir or current dir.

        """
        super().__init__()

        directory = directory or get_run_directory()
        directory = Path(directory)
        self.directory = directory
        if (directory / self._filename).is_file():
            if overwrite:
                logger.info(f"Found previous run. Removing '{directory}'.")
                for root, dirs, files in os.walk(directory):
                    for f in files:
                        full_fn = os.path.join(root, f)
                        if ".hydra" not in full_fn:
                            os.remove(full_fn)
                            logger.debug(f"Removed {full_fn}")
            else:
                raise RuntimeError(
                    f"Found previous run at '{directory}'. Stopping run. If you want to overwrite, specify overwrite "
                    f"for the file logger in the config (CARP-S/carps/configs/logger.yaml).")

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
        info = convert_trials(n_trials, trial_info, trial_value)
        if logging.DEBUG <= logger.level:
            info_str = json.dumps(info) + "\n"
            logger.debug(info_str)
        else:
            info_str = f"n_trials: {info['n_trials']}, config: {info['trial_info']['config']}, cost: {info['trial_value']['cost']}"
            logger.info(info_str)

        dump_logs(log_data=info, filename=self._filename, directory=self.directory)

    def log_incumbent(self, n_trials: int, incumbent: Incumbent) -> None:
        if incumbent is None:
            return
        if not isinstance(incumbent, list):
            incumbent = [incumbent]

        for inc in incumbent:
            info = convert_trials(n_trials, inc[0], inc[1])
            dump_logs(log_data=info, filename=self._filename_trajectory, directory=self.directory)

    def log_arbitrary(self, data: dict, entity: str) -> None:
        dump_logs(log_data=data, filename=f"{entity}.jsonl", directory=self.directory)
