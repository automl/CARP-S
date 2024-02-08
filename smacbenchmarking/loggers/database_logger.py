from __future__ import annotations

import json
from dataclasses import asdict

from smacbenchmarking.database.result_processor import ResultProcessor
from smacbenchmarking.loggers.abstract_logger import AbstractLogger
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class DatabaseLogger(AbstractLogger):
    def __init__(self, result_processor: ResultProcessor) -> None:
        super().__init__()
        self.result_processor = result_processor

    def log_trial(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        info = {"trial_info": asdict(trial_info), "trial_value": asdict(trial_value)}
        info["trial_info"]["config"] = json.dumps(asdict(trial_info)['config'].get_dictionary())
        info["trial_value"]["status"] = info["trial_value"]["status"].name
        info["trial_value"]["additional_info"] = json.dumps(info["trial_value"]["additional_info"])
        info["trial_value"]["cost"] = json.dumps({'cost': json.dumps(info["trial_value"]["cost"])})
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

        info_2 = {
            'trial_info__config': info['trial_info__config'],
            'trial_info__instance': info['trial_info__instance'] if 'trial_info__instance' in info else None,
            'trial_info__seed': info['trial_info__seed'] if 'trial_info__seed' in info else None,
            'trial_info__budget': info['trial_info__budget'] if 'trial_info__budget' in info else None,
            'trial_value__cost': info['trial_info__cost'],
            'trial_value__time': info['trial_value__time'],
            'trial_value__status': info['trial_value__status'],
            'trial_value__starttime': info['trial_value__starttime'],
            'trial_value__endtime': info['trial_value__endtime'],
            'trial_value__additional_info': info['trial_value__additional_info'] if 'trial_value__additional_info' in info else None,
        }

        print(log)

        self.result_processor.process_logs({"trials": info_2})
