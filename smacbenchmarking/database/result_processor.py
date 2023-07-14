from __future__ import annotations
from py_experimenter.database_connector import DatabaseConnector
from py_experimenter.result_processor import ResultProcessor as PyExpResultProcessor
from smacbenchmarking.database.connector import DatabaseConnectorMySQL

from omegaconf import DictConfig


class ResultProcessor(PyExpResultProcessor):
    def __init__(self, config: DictConfig, use_codecarbon: bool, codecarbon_config: DictConfig, database_cfg, table_name: str, result_fields: list[str], experiment_id: int):
        self._table_name = table_name
        self._result_fields = result_fields
        self._config = config
        self._timestamp_on_result_fields = True  # TODO utils.timestamps_for_result_fields(self._config)
        self._experiment_id = experiment_id
        self._experiment_id_condition = f'ID = {self._experiment_id}'

        self.use_codecarbon = use_codecarbon
        self._codecarbon_config = codecarbon_config

        self._dbconnector: DatabaseConnector = DatabaseConnectorMySQL(database_cfg=database_cfg)
        self._dbconnector.config = config
        self._dbconnector.create_table_if_not_existing()
