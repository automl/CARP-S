"""Reset experiments that have errored out in the database."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import fire
from omegaconf import OmegaConf
from py_experimenter.experiment_status import ExperimentStatus
from py_experimenter.experimenter import PyExperimenter

if TYPE_CHECKING:
    from py_experimenter.database_connector_mysql import DatabaseConnectorMYSQL

YAHPO_ERROR_CONDITION = r"WHERE `benchmark_id` LIKE 'YAHPO' AND `status` LIKE 'error' AND `error` LIKE '%AttributeError: \'NoneType\' object has no attribute \'update\'%'"  # noqa: E501


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
    column_names, entries = database_connector._get_experiments_with_condition(condition)
    database_connector._delete_experiments_with_condition(condition)
    return column_names, entries


def main(
    reset_yahpo_attr_error: bool = False,  # noqa: FBT001, FBT002
    pyexperimenter_configuration_file_path: str | None = None,
    database_credential_file_path: str | Path | None = None,
) -> None:
    """Reset experiments that have errored out in the database.

    Usage: `python -m carps.experimenter.database.reset_experiments` for resetting all experiments with status error or
    `python -m carps.experimenter.database.reset_experiments --reset_yahpo_attr_error` for resetting experiments with
    this specific yahpo error condition.

    Args:
        reset_yahpo_attr_error (bool, optional): If True, reset only experiments with the YAHPO error condition.
            Defaults to False.
        pyexperimenter_configuration_file_path (str, optional): Path to the py_experimenter configuration file.
            Defaults to None.
        database_credential_file_path (str | Path, optional): Path to the database credential file. Defaults to None.
    """
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
    if not reset_yahpo_attr_error:
        experimenter.db_connector.reset_experiments(ExperimentStatus.ERROR.value)
    else:
        reset_experiments_with_condition(experimenter.db_connector, YAHPO_ERROR_CONDITION)


if __name__ == "__main__":
    fire.Fire(main)
