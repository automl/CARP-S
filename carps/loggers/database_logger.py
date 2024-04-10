from __future__ import annotations

import json
from dataclasses import asdict

from py_experimenter.result_processor import ResultProcessor

from carps.loggers.abstract_logger import AbstractLogger
from carps.optimizers.optimizer import Incumbent
from carps.utils.trials import TrialInfo, TrialValue


class DatabaseLogger(AbstractLogger):

    def __init__(self, result_processor: ResultProcessor) -> None:
        super().__init__()
        self.result_processor = result_processor

    def log_trial(self, n_trials: int, trial_info: TrialInfo, trial_value: TrialValue) -> None:
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

        info["n_trials"] = n_trials

        self.result_processor.process_logs({"trials": info})

    def log_incumbent(self, incumbent: Incumbent) -> None:
        pass

    def log_arbitrary(self, data: dict, entity: str) -> None:
        self.result_processor.process_logs({entity: data})
