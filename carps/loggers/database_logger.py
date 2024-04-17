from __future__ import annotations

import json
import logging
from dataclasses import asdict

from py_experimenter.result_processor import ResultProcessor
from rich.logging import RichHandler
from smac.utils.logging import get_logger

from carps.loggers.abstract_logger import AbstractLogger
from carps.optimizers.optimizer import Incumbent
from carps.utils.trials import TrialInfo, TrialValue

FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = get_logger("DatabaseLogger")


def convert_trial_info(trial_info, trial_value):
    info = {"trial_info": asdict(trial_info), "trial_value": asdict(trial_value)}
    info["trial_info"]["config"] = json.dumps(asdict(trial_info)['config'].get_dictionary())
    info["trial_value"]["status"] = info["trial_value"]["status"].name
    info["trial_value"]["additional_info"] = json.dumps(info["trial_value"]["additional_info"])
    info["trial_value"]["cost"] = json.dumps({'cost': json.dumps(info["trial_value"]["cost"])})
    keys = ["trial_info", "trial_value"]
    for key in keys:
        d = info.pop(key)
        for k, v in d.items():
            if v is not None:
                k_new = f"{key}__{k}"
                info[k_new] = v
            else:
                # If v is None, we omit it from the dict.
                # Missing keys will automatically be filled with NULL in MySQL.
                pass
    return info


class DatabaseLogger(AbstractLogger):

    def __init__(self, result_processor: ResultProcessor | None = None) -> None:
        super().__init__()
        self.result_processor = result_processor
        if self.result_processor is None:
            logger.info("Not logging to database (result processor is None).")

    def log_trial(self, n_trials: int, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        info = convert_trial_info(trial_info, trial_value)
        info["n_trials"] = n_trials

        if self.result_processor:
            self.result_processor.process_logs({"trials": info})

    def log_incumbent(self, n_trials: int, incumbent: Incumbent) -> None:
        if incumbent is None:
            return

        if not isinstance(incumbent, list):
            incumbent = [incumbent]

        for inc in incumbent:
            info = convert_trial_info(inc[0], inc[1])
            info["n_trials"] = n_trials

            if self.result_processor:
                self.result_processor.process_logs({"trajectory": info})

    def log_arbitrary(self, data: dict, entity: str) -> None:
        if self.result_processor:
            self.result_processor.process_logs({entity: data})
