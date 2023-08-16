from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from omegaconf import DictConfig
from rich import print as printr
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.database import utils
from smacbenchmarking.database.connector import DatabaseConnectorMySQL
from smacbenchmarking.database.result_processor import ResultProcessor
from smacbenchmarking.database.utils import load_config
from smacbenchmarking.loggers.abstract_logger import AbstractLogger


class DatabaseLogger(AbstractLogger):
    def __init__(self, problem: Problem, cfg: DictConfig) -> None:
        super().__init__(problem, cfg)
        self.buffer: list[dict] = []
        connector = DatabaseConnectorMySQL(database_cfg=cfg.database)
        connector.config = cfg
        connector.create_table_if_not_existing()
        parameters = utils.get_keyfield_data(cfg)
        connector.config = cfg
        connector.fill_table(parameters=parameters)
        printr("Databse parameters:", parameters)
        experiment_id = connector.find_experiment_id(parameters)
        printr("Database experiment id:", experiment_id)

        table_name = cfg.database.table_name
        self.result_processor = ResultProcessor(
            config=cfg,
            use_codecarbon=False,
            table_name=table_name,
            codecarbon_config=None,
            database_cfg=cfg.database,
            experiment_id=experiment_id,  # The AUTO_INCREMENT is 1-based
            result_fields=cfg["database"]["resultfields"],
        )
        print("Created tables")

    def trial_to_buffer(self, trial_info: TrialInfo, trial_value: TrialValue, n_trials: int | None = None) -> None:
        info = {"n_trials": n_trials, "trial_info": asdict(trial_info), "trial_value": asdict(trial_value)}
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
        #             "n_trials": self.n_trials,
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
        self.buffer.append(log)

    def write_buffer(self) -> None:
        if self.buffer:
            for log in self.buffer:
                self.result_processor.process_logs(log)
            self.buffer = []
