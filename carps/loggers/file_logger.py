from __future__ import annotations

import json
import logging
import os
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode

from carps.loggers.abstract_logger import AbstractLogger
from carps.utils.loggingutils import get_logger, setup_logging, CustomEncoder

if TYPE_CHECKING:
    from carps.optimizers.optimizer import Incumbent
    from carps.utils.trials import TrialInfo, TrialValue

setup_logging()
logger = get_logger("FileLogger")


def get_run_directory() -> Path:
    """Get current run dir.

    Either '.' if hydra not active, else hydra run dir.

    Returns:
    -------
    Path
        Directory
    """
    try:
        # Check if we are in a hydra context
        hydra_cfg = HydraConfig.instance().get()
        if hydra_cfg.mode == RunMode.RUN:
            return Path(hydra_cfg.run.dir)

        # TODO: How can we check to actually make sure it's a multi-run...
        # MULTIRUN
        return Path(hydra_cfg.sweep.dir) / hydra_cfg.sweep.subdir
    except Exception as e:
        cwd = Path.cwd()
        tb = traceback.format_exc()
        msg = f"Unexpected issue getting current run_directory!\n{tb}\n{e}\n\nReturning current directory of {cwd}."
        warnings.warn(msg, category=UserWarning, stacklevel=2)
        return cwd


def dump_logs(log_data: dict, filename: str, directory: str | Path | None = None):
    """Dump log dict in jsonl format.

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
    log_data_str = json.dumps(log_data, cls=CustomEncoder) + "\n"
    _dir = Path(directory) if directory is not None else get_run_directory()
    filepath = _dir / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("a") as file:
        file.writelines([log_data_str])


def convert_trials(n_trials, trial_info, trial_value, n_function_calls: int | None = None):
    if n_function_calls is None:
        n_function_calls = n_trials
    info = {
        "n_trials": n_trials,
        "n_function_calls": n_function_calls,
        "trial_info": asdict(trial_info),
        "trial_value": asdict(trial_value),
    }
    info["trial_info"]["config"] = list(dict(info["trial_info"]["config"]).values())
    info["trial_value"]["virtual_time"] = float(info["trial_value"]["virtual_time"])
    return info


class FileLogger(AbstractLogger):
    _filename: str = "trial_logs.jsonl"
    _filename_trajectory: str = "trajectory_logs.jsonl"

    def __init__(self, overwrite: bool = False, directory: str | Path | None = None) -> None:
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

        directory = Path(directory) if directory is not None else get_run_directory()
        self.directory = directory
        if (directory / self._filename).is_file():
            if overwrite:
                logger.info(f"Found previous run. Removing '{directory}'.")
                for root, _dirs, files in os.walk(directory):
                    for f in files:
                        full_fn = os.path.join(root, f)
                        if ".hydra" not in full_fn:
                            os.remove(full_fn)
                            logger.debug(f"Removed {full_fn}")
            else:
                raise RuntimeError(
                    f"Found previous run at '{directory}'. Stopping run. If you want to overwrite, specify overwrite "
                    f"for the file logger in the config (CARP-S/carps/configs/logger.yaml)."
                )

    def log_trial(
        self, n_trials: int, trial_info: TrialInfo, trial_value: TrialValue, n_function_calls: int | None = None
    ) -> None:
        """Evaluate the problem and log the trial.

        Parameters
        ----------
        n_trials : float
            The number of trials that have been run so far.
            For the case of multi-fidelity, a full trial
            is a configuration evaluated on the maximum budget and
            the counter is increased by `budget/max_budget` instead
            of 1.
        trial_info : TrialInfo
            The trial info.
        trial_value : TrialValue
            The trial value.
        n_function_calls: int | None, default None
            The number of target function calls, no matter the budget.
        """
        info = convert_trials(n_trials, trial_info, trial_value, n_function_calls)
        if logger.level >= logging.DEBUG:
            info_str = json.dumps(info, cls=CustomEncoder) + "\n"
            logger.debug(info_str)
        else:
            info_str = f"n_trials: {info['n_trials']}, config: {info['trial_info']['config']}, cost: {info['trial_value']['cost']}"
            if info["trial_info"]["budget"] is not None:
                info_str += f" budget: {info['trial_info']['budget']}"
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
