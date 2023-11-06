from __future__ import annotations

import json
from dataclasses import asdict

from omegaconf import DictConfig

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.loggers.abstract_logger import AbstractLogger

import logging
from pathlib import Path


from smacbenchmarking.utils.trials import TrialInfo, TrialValue

import json
from hydra.core.hydra_config import HydraConfig
from pathlib import Path




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
        hydra_cfg = HydraConfig.instance().get()
        directory = hydra_cfg.run.dir
    except:
        directory = "."  # TODO fix directory
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

        # self.logger = logging.getLogger("filelogger")
        # formatter = logging.Formatter('%(message)s')
        # # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        # fh = logging.FileHandler('trial_logs.jsonl')
        # fh.setLevel(logging.DEBUG)
        # fh.setFormatter(formatter)
        # self.logger.addHandler(fh)


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

        dump_logs(log_data=info, filename="trial_logs.jsonl")
