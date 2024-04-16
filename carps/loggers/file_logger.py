from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from rich.logging import RichHandler
from smac.utils.logging import get_logger

from carps.loggers.abstract_logger import AbstractLogger
from carps.optimizers.optimizer import Incumbent
from carps.utils.trials import TrialInfo, TrialValue

FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

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


class FileLogger(AbstractLogger):
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
        if (directory / "trial_logs.jsonl").is_file():
            if overwrite:
                logger.info(f"Found previous run. Removing '{directory}'.")
                for root, dirs, files in os.walk(directory):
                    for f in files:
                        full_fn = os.path.join(root, f)
                        if ".hydra" not in full_fn:
                            os.remove(full_fn)
                            logger.debug(f"Removed {full_fn}")
            else:
                raise RuntimeError(f"Found previous run at '{directory}'. Stopping run. If you want to overwrite, specify overwrite for the file logger in the config (CARP-S/carps/configs/logger.yaml).")
                   

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

        dump_logs(log_data=info, filename="trial_logs.jsonl", directory=self.directory)

    def log_incumbent(self, incumbent: Incumbent) -> None:
        pass

    def log_arbitrary(self, data: dict, entity: str) -> None:
        dump_logs(log_data=data, filename=f"{entity}.jsonl", directory=self.directory)
