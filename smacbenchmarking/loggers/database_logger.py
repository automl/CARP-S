from __future__ import annotations

from dataclasses import asdict

from omegaconf import DictConfig

from smacbenchmarking.database.result_processor import ResultProcessor
from smacbenchmarking.loggers.abstract_logger import AbstractLogger
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class DatabaseLogger(AbstractLogger):
    def __init__(self, result_processor: ResultProcessor) -> None:
        super().__init__()
        self.result_processor = result_processor

    def log_trial(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        info = {"trial_info": asdict(trial_info), "trial_value": asdict(trial_value)}
        info["trial_info"]["config"] = str(
            list(dict(info["trial_info"]["config"]).values())
        )  # TODO beautify serialization
        info["trial_value"]["status"] = info["trial_value"]["status"].name
        info["trial_value"]["additional_info"] = str(info["trial_value"]["additional_info"])
        keys = ["trial_info", "trial_value"]
        for key in keys:
            D = info.pop(key)
            for k, v in D.items():
                if v is not None:
                    k_new = f"{key}__{k}"
                    info[k_new] = v
                else:
                    # If v is None, we omit it from the dict.
                    # Missing keys will automatically filled with NULL in MySQL.
                    pass

        log = {"trials": info}

        # log = {
        #     "trials":
        #         {
        #             "trial_info__config": str([0, 1]),
        #             "trial_info__instance": 0,
        #             "trial_info__seed": 0,
        #             "trial_info__budget": 0,
        #             "trial_value__cost": str(100),
        #             "trial_value__time": 3,
        #             "trial_value__starttime": 12345,
        #             "trial_value__starttime": 12348,
        #             "trial_value__status": "OK",
        #         }
        # }
        self.result_processor.process_logs(log)
