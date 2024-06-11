from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from carps.loggers.abstract_logger import AbstractLogger
from carps.utils.loggingutils import get_logger, setup_logging, CustomEncoder

if TYPE_CHECKING:
    from py_experimenter.result_processor import ResultProcessor

    from carps.optimizers.optimizer import Incumbent
    from carps.utils.trials import TrialInfo, TrialValue

setup_logging()
logger = get_logger("DatabaseLogger")


def convert_trial_info(trial_info, trial_value):
    info = {"trial_info": asdict(trial_info), "trial_value": asdict(trial_value)}
    info["trial_info"]["config"] = json.dumps(asdict(trial_info)["config"].get_dictionary(), cls=CustomEncoder)
    info["trial_value"]["status"] = info["trial_value"]["status"].name
    info["trial_value"]["additional_info"] = json.dumps(info["trial_value"]["additional_info"], cls=CustomEncoder)
    info["trial_value"]["cost"] = json.dumps({"cost": json.dumps(info["trial_value"]["cost"], cls=CustomEncoder)}, cls=CustomEncoder)
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
        info = convert_trial_info(trial_info, trial_value)
        info["n_trials"] = n_trials
        info["n_function_calls"] = n_function_calls if n_function_calls else n_trials

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
