"""Reset experiments that have errored out in the database.

Useful commands to copy and paste:
- `python -m carps.experimenter.database.reset_experiments` for resetting all experiments with status error
- `python -m carps.experimenter.database.reset_experiments ['yahpo_attr_error']` for resetting experiments with
  this specific yahpo error condition.
- `python -m carps.experimenter.database.reset_experiments ['falsely_done']` for resetting experiments that are
  falsely marked as done (experiment ids not present in trials table).
- `python -m carps.experimenter.database.reset_experiments ['error', 'yahpo_attr_error', 'falsely_done']` for
    resetting all.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import fire
import pandas as pd
from omegaconf import OmegaConf
from py_experimenter.experiment_status import ExperimentStatus
from py_experimenter.experimenter import PyExperimenter

if TYPE_CHECKING:
    from py_experimenter.database_connector_mysql import DatabaseConnectorMYSQL

YAHPO_ERROR_CONDITION = r"WHERE `benchmark_id` LIKE 'YAHPO' AND `status` LIKE 'error' AND `error` LIKE '%AttributeError: \'NoneType\' object has no attribute \'update\'%'"  # noqa: E501
FALSELY_DONE_CONDITION_SELECT = r"SELECT r.* FROM `results` r LEFT JOIN `results__trials` rt ON r.`ID` = rt.`experiment_id` WHERE rt.`experiment_id` IS NULL AND r.`status` = 'done';"  # noqa: E501
FALSELY_DONE_CONDITION_DELETE = r"DELETE r FROM `results` r LEFT JOIN `results__trials` rt ON r.`ID` = rt.`experiment_id` WHERE rt.`experiment_id` IS NULL AND r.`status` = 'done';"  # noqa: E501


def reset_experiments_with_condition(database_connector: DatabaseConnectorMYSQL, condition: str) -> None:
    """Reset experiments with a specific condition in the database.

    Args:
        database_connector (DatabaseConnectorMYSQL): The database connector instance.
        condition (str): The condition to filter experiments for resetting.
    """

    def get_dict_for_keyfields_and_rows(keyfields: list[str], rows: list[list[str]]) -> list[dict]:
        return [dict(zip(keyfields, row, strict=True)) for row in rows]

    keyfields, rows = pop_experiments_with_condition(database_connector, condition)
    row_dicts = get_dict_for_keyfields_and_rows(keyfields, rows)
    if row_dicts:
        database_connector.fill_table(row_dicts)
    database_connector.logger.info(f"{len(row_dicts)} experiments with condition {condition} were reset")


def reset_falsely_done_experiments(database_connector: DatabaseConnectorMYSQL) -> None:
    """Reset experiments that are falsely marked as done in the database.

    Experiments, that are falsely done, fulfill following condition:
    They are marked as done in the results table, but do not have a corresponding entry in the trials table.

    Args:
        database_connector (DatabaseConnectorMYSQL): The database connector instance.
    """

    def get_dict_for_keyfields_and_rows(keyfields: list[str], rows: list[list[str]]) -> list[dict]:
        return [dict(zip(keyfields, row, strict=True)) for row in rows]

    column_names, entries = get_experiments_with_condition(database_connector, FALSELY_DONE_CONDITION_SELECT)
    delete_experiments_with_condition(database_connector, FALSELY_DONE_CONDITION_DELETE)
    row_dicts = get_dict_for_keyfields_and_rows(column_names, entries)
    if row_dicts:
        database_connector.fill_table(row_dicts)
    database_connector.logger.info(
        f"{len(row_dicts)} experiments with condition {FALSELY_DONE_CONDITION_SELECT} were reset"
    )


def get_experiments_with_condition(
    database_connector: DatabaseConnectorMYSQL, condition: str | None = None
) -> tuple[list[str], list[list]]:
    """Get experiments with a specific condition from the database.

    Args:
        database_connector (DatabaseConnectorMYSQL): The database connector instance.
        condition (str | None): The condition to filter experiments. Defaults to None.

    Returns:
        tuple[list[str], list[list]]: A tuple containing the column names and the entries of the experiments.
    """

    def _get_keyfields_from_columns(column_names: list[str], entries: list[dict]) -> tuple[list[str], list[list]]:
        df = pd.DataFrame(entries, columns=column_names)  # noqa: PD901
        keyfields = database_connector.database_configuration.keyfields.keys()
        entries = df[keyfields].values.tolist()  # noqa: PD011
        return keyfields, entries  # type: ignore[return-value]

    connection = database_connector.connect()
    cursor = database_connector.cursor(connection)

    query_condition = condition or ""
    if "SELECT" not in query_condition:
        query = f"SELECT * FROM {database_connector.database_configuration.table_name} {query_condition}"  # noqa: S608
    else:
        query = query_condition
    database_connector.execute(cursor, query)
    entries = database_connector.fetchall(cursor)
    column_names = database_connector.get_structure_from_table(cursor)
    column_names, entries = _get_keyfields_from_columns(column_names, entries)

    return column_names, entries


def delete_experiments_with_condition(
    database_connector: DatabaseConnectorMYSQL, delete_query: str | None = None
) -> None:
    """Delete experiments with a specific condition from the database.

    Args:
        database_connector (DatabaseConnectorMYSQL): The database connector instance.
        delete_query (str | None): The condition to filter experiments for deletion. Defaults to None.
    """
    connection = database_connector.connect()
    cursor = database_connector.cursor(connection)

    query_condition = delete_query or ""
    if "DELETE" not in query_condition:
        query = f"DELETE FROM {database_connector.database_configuration.table_name} {query_condition}"  # noqa: S608
    else:
        query = query_condition
    database_connector.execute(cursor, query)
    database_connector.commit(connection)
    database_connector.close_connection(connection)


def pop_experiments_with_condition(
    database_connector: DatabaseConnectorMYSQL, condition: str | None = None
) -> tuple[list[str], list[list]]:
    """Pop experiments with a specific condition from the database.

    Args:
        database_connector (DatabaseConnectorMYSQL): The database connector instance.
        condition (str | None): The condition to filter experiments for popping. Defaults to None.

    Returns:
        tuple[list[str], list[list]]: A tuple containing the column names and the entries of the popped experiments.
    """
    column_names, entries = get_experiments_with_condition(database_connector, condition)
    delete_experiments_with_condition(database_connector, condition)
    return column_names, entries


def main(
    reset_what: list[str] | None = None,
    pyexperimenter_configuration_file_path: str | None = None,
    database_credential_file_path: str | Path | None = None,
) -> None:
    """Reset experiments that have errored out in the database.

    Usage: `python -m carps.experimenter.database.reset_experiments` for resetting all experiments with status error or
    `python -m carps.experimenter.database.reset_experiments --reset_yahpo_attr_error` for resetting experiments with
    this specific yahpo error condition.

    Args:
        reset_what (list[str] | None, optional): List of conditions to reset. Defaults to None.
            Valid options are: "error", "yahpo_attr_error", "falsely_done".
            "error": Reset all errored experiments.
            "yahpo_attr_error": Reset experiments with the YAHPO error condition.
            "falsely_done": Reset experiments that are falsely marked as done (experiment ids not present in trials
                table).
        pyexperimenter_configuration_file_path (str, optional): Path to the py_experimenter configuration file.
            Defaults to None.
        database_credential_file_path (str | Path, optional): Path to the database credential file. Defaults to None.
    """
    if reset_what is None:
        reset_what = ["error"]

    experiment_configuration_file_path = (
        pyexperimenter_configuration_file_path
        or Path(__file__).parent.parent.parent / "experimenter/py_experimenter.yaml"
    )

    database_credential_file_path = (
        database_credential_file_path or Path(__file__).parent.parent.parent / "experimenter/credentials.yaml"
    )
    if database_credential_file_path is not None and not Path(database_credential_file_path).exists():
        database_credential_file_path = None

    experimenter = PyExperimenter(
        experiment_configuration_file_path=experiment_configuration_file_path,
        name="remove_error_rows",
        database_credential_file_path=database_credential_file_path,
        log_file="logs/reset_experiments.log",
        use_ssh_tunnel=OmegaConf.load(experiment_configuration_file_path).PY_EXPERIMENTER.Database.use_ssh_tunnel,
    )
    for reset_this in reset_what:
        if reset_this == "error":
            experimenter.db_connector.reset_experiments(ExperimentStatus.ERROR.value)
        elif reset_this == "yahpo_attr_error":
            reset_experiments_with_condition(experimenter.db_connector, YAHPO_ERROR_CONDITION)
        elif reset_this == "falsely_done":
            reset_falsely_done_experiments(experimenter.db_connector)
        else:
            raise ValueError(
                f"Unknown reset condition: {reset_this}. Valid options are: " "error, yahpo_attr_error, falsely_done."
            )


if __name__ == "__main__":
    fire.Fire(main)
