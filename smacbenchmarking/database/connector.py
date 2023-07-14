from __future__ import annotations

from typing import Any
from py_experimenter.database_connector_mysql import DatabaseConnectorMYSQL
from omegaconf import OmegaConf, DictConfig
from rich import print as printr
from smacbenchmarking.database import utils
import logging
import pandas as pd
import warnings
from py_experimenter.exceptions import (CreatingTableError, DatabaseConnectionError, EmptyFillDatabaseCallError, NoExperimentsLeftException,
                                        TableHasWrongStructureError)
from py_experimenter.experiment_status import ExperimentStatus
from smacbenchmarking.database.utils import load_config


class DatabaseConnectorMySQL(DatabaseConnectorMYSQL):
    def __init__(self, database_cfg: str | DictConfig, use_codecarbon: bool = False):
        if type(database_cfg) == str:
            database_cfg = load_config(database_cfg)
        
        self.host = database_cfg["host"]
        self.user = database_cfg["user"]
        self.password = database_cfg["password"]

        self.table_name = database_cfg["table_name"]
        self.database_name = database_cfg["database_name"]
        self.logtableaddition = database_cfg.get("logtableaddition", None)

        self.database_credentials = self._extract_credentials()
        self.timestamp_on_result_fields = True  # TODO:utils.timestamps_for_result_fields(self.config)

        self.use_codecarbon = use_codecarbon
        self._test_connection()
        self._create_database_if_not_existing()

    def fill_table(self, parameters=None, fixed_parameter_combinations=None) -> None:
        logging.debug("Fill table with parameters.")
        parameters = parameters if parameters is not None else {}
        fixed_parameter_combinations = fixed_parameter_combinations if fixed_parameter_combinations is not None else []

        keyfield_names = utils.get_keyfield_names(self.config)
        combinations = utils.combine_fill_table_parameters(keyfield_names, parameters, fixed_parameter_combinations)

        if len(combinations) == 0:
            raise EmptyFillDatabaseCallError("No combinations to execute found.")

        column_names = list(combinations[0].keys())
        logging.debug("Getting existing rows.")
        existing_rows = set(self._get_existing_rows(column_names))
        time = utils.get_timestamp_representation()

        rows_skipped = 0
        rows = []
        logging.debug("Checking which of the experiments to be inserted already exist.")
        for combination in combinations:
            if self._check_combination_in_existing_rows(combination, existing_rows, keyfield_names):
                rows_skipped += 1
                continue
            values = list(combination.values())
            values.append(ExperimentStatus.CREATED.value)
            values.append(time)
            rows.append(values)

        if rows:
            logging.debug(f"Now adding {len(rows)} rows to database. {rows_skipped} rows were skipped.")
            self._write_to_database(rows, column_names + ["status", "creation_date"])
            logging.info(f"{len(rows)} rows successfully added to database. {rows_skipped} rows were skipped.")
        else:
            logging.info(f"No rows to add. All the {len(combinations)} experiments already exist.")


    def _exclude_fixed_columns(self, columns: list[str]) -> list[str]:
        amount_of_keyfields = len(utils.get_keyfield_names(self.config))
        amount_of_result_fields = len(utils.get_result_field_names(self.config))

        if self.timestamp_on_result_fields:
            amount_of_result_fields *= 2

        return columns[1:amount_of_keyfields + 1] + columns[-amount_of_result_fields - 2:-2]
    
    def _execute_queries(self, connection, cursor) -> tuple[int, list, list]:
        order_by = "id"
        time = utils.get_timestamp_representation()

        self.execute(cursor, f"SELECT id FROM {self.table_name} WHERE status = 'created' ORDER BY {order_by} LIMIT 1;")
        experiment_id = self.fetchall(cursor)[0][0]
        self.execute(
            cursor, f"UPDATE {self.table_name} SET status = {self._prepared_statement_placeholder}, start_date = {self._prepared_statement_placeholder} WHERE id = {self._prepared_statement_placeholder};", (ExperimentStatus.RUNNING.value, time, experiment_id))
        keyfields = ','.join(utils.get_keyfield_names(self.config))
        self.execute(cursor, f"SELECT {keyfields} FROM {self.table_name} WHERE id = {experiment_id};")
        values = self.fetchall(cursor)
        self.commit(connection)
        description = cursor.description
        return experiment_id, description, values
    
    def find_experiment_id(self, parameters: dict[str, Any]) -> int:
        connection = self.connect()
        cursor = self.cursor(connection)

        key_exp_id = "id"
        param_statement = []
        for k, v in parameters.items():
            if type(v) == str:
                v = f"'{v}'"
            string = f"{k} = {v}"
            if v is None:
                string = f"{k} IS NULL"            
            param_statement.append(string)
        param_statement = " AND ".join(param_statement)
        
        query = f"SELECT {key_exp_id} FROM {self.table_name} WHERE {param_statement};"
        self.execute(cursor, query)
        values = self.fetchall(cursor)[0]
        self.commit(connection)
        description = cursor.description
        ids = values
        if len(ids) > 1:
            warnings.warn("Something is weird. Got multiple experiment ids for the same parameters. "
                          f"{ids} from query {query}.")
        return ids[0]
    
    def create_table_if_not_existing(self) -> None:
        logging.debug("Create table if not exist")

        keyfields = utils.get_keyfields(self.config)
        resultfields = utils.get_resultfields(self.config)
        if self.timestamp_on_result_fields:
            resultfields = utils.add_timestep_result_columns(resultfields)

        connection = self.connect()
        cursor = self.cursor(connection)
        if self._table_exists(cursor):
            if not self._table_has_correct_structure(cursor, keyfields + resultfields):
                raise TableHasWrongStructureError("Keyfields or resultfields from the configuration do not match columns in the existing "
                                                  "table. Please change your configuration or delete the table in your database.")
        else:
            columns = self._compute_columns(keyfields, resultfields)
            self._create_table(cursor, columns, self.table_name, table_type="standard")

            cfg = self.config
            logtable_cfg = OmegaConf.to_container(cfg.database.logtables, resolve=True)
            flattened_logtable_cfg = pd.json_normalize(logtable_cfg, sep="__").iloc[0].to_dict()

            log_table_name = f"{self.table_name}__trials"

            columns = [(k, v) for k, v in flattened_logtable_cfg.items()]
            self._create_table(cursor, columns=columns, table_name=log_table_name, table_type="logtable", logatableaddition=self.logtableaddition)

    def _create_table(self, cursor, columns: list[tuple['str']], table_name: str, table_type: str = 'standard', logatableaddition: str | None = None):
        query = self._get_create_table_query(columns, table_name, table_type, logatableaddition=logatableaddition)
        try:
            self.execute(cursor, query)
        except Exception as err:
            raise CreatingTableError(f'Error when creating table: {err}')

    def _get_create_table_query(self, columns: list[tuple['str']], table_name: str, table_type: str = 'standard', logatableaddition: str | None = None):
        columns = ['%s %s DEFAULT NULL' % (field, datatype) for field, datatype in columns]
        columns = ','.join(columns)
        query = f"CREATE TABLE {table_name} (ID INTEGER PRIMARY KEY {self.get_autoincrement()}"
        if table_type == 'standard':
            query += f", {columns}"
        elif table_type == 'logtable':
            query += f", experiment_id INTEGER, timestamp DATETIME, {columns}, FOREIGN KEY (experiment_id) REFERENCES {self.table_name}(ID) ON DELETE CASCADE"
            if logatableaddition:
                query += f", {logatableaddition}"
        elif table_type == 'codecarbon':
            query += f", experiment_id INTEGER, {columns}, FOREIGN KEY (experiment_id) REFERENCES {self.table_name}(ID) ON DELETE CASCADE"
        else:
            raise ValueError(f"Unknown table type: {table_type}")
        return query + ');'
