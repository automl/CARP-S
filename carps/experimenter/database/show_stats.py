"""Reset experiments that have errored out in the database."""

from __future__ import annotations

import re
from pathlib import Path

import fire
import pandas as pd
from omegaconf import OmegaConf
from py_experimenter.experiment_status import ExperimentStatus
from py_experimenter.experimenter import PyExperimenter

from carps.utils.loggingutils import get_logger, setup_logging

setup_logging()
logger = get_logger("DB Stats")


def main(
    pyexperimenter_configuration_file_path: str | None = None, database_credential_file_path: str | Path | None = None
) -> None:
    """Show status of experiments in the database. Export error info.

    Args:
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
        name="show_stats",
        database_credential_file_path=database_credential_file_path,
        log_file="logs/reset_experiments.log",
        use_ssh_tunnel=OmegaConf.load(experiment_configuration_file_path).PY_EXPERIMENTER.Database.use_ssh_tunnel,
    )

    for status in ExperimentStatus:
        if status == ExperimentStatus.ALL:
            continue
        column_names, entries = experimenter.db_connector._get_experiments_with_condition(
            condition=f"WHERE status = '{status.value}'"
        )
        # column names most likely are ['config', 'config_hash', 'benchmark_id', 'task_id', 'subset_id', 'task_type',
        # 'optimizer_id', 'optimizer_container_id', 'seed', 'n_trials', 'time_budget']
        logger.info(f"Number of experiments with status {status.value}: {len(entries)}")
        if status == ExperimentStatus.ERROR:
            error_info = []
            for entry in entries:
                error_dict = dict(zip(list(column_names)[2:], entry[2:], strict=True))
                error_info.append(error_dict)
            error_info = pd.DataFrame(error_info)
            error_info.to_csv("error_info.csv", index=False)  # type: ignore[attr-defined]
            logger.info(
                "Error info saved to error_info.csv. Check this and see whether you can fix the issues."
                " When done, reset those experiments with `python -m carps.experimenter.database.reset_experiments`."
            )

    # Get error rows with error message
    # Split into unknown errors and the typical yahpo error. The latter experiments can be easily reset and rerun.
    table = experimenter.db_connector.get_table()
    exclude_keys = ["config", "config_hash"]
    error_status = "error"
    error_rows = table[table["status"] == error_status]
    error_rows = error_rows.drop(columns=exclude_keys)

    expected_errors = {
        "yahpo_localconfig": "Exception('Could not load local_config! Please run LocalConfiguration.init_config() "
        "and restart.')",
        # Reset with `python -m carps.experimenter.database.reset_experiments --reset_yahpo_attr_error`
        # Just rerun those, it is due to multiple accesses of the surrogate model
        "yahpo_updatenone": "AttributeError: 'NoneType' object has no attribute 'update'",
        "smac_intensifier": "assert len(incumbent_isb_keys) > 0",
        # python -m carps.run +task/subselection/blackbox/dev=subset_yahpo_rbv2_aknn_1462_None +optimizer/synetune=KDE seed=3  # noqa: E501
        "synetune_does_not_respect_bounds": "ConfigSpace.exceptions.IllegalValueError: Value 5: (<class 'int'>) is not allowed for hyperparameter with name 'M'",  # noqa: E501
    }
    logger.info(f"Expected errors: {expected_errors.keys()}")
    known_error_ids = []
    for error_id, error_msg in expected_errors.items():
        error_ids = error_rows["error"].str.contains(error_msg, regex=False)
        error_rows[error_ids].to_csv(f"error_{error_id}.csv", index=False)
        known_error_ids.append(error_ids)
        logger.info(f"Number of experiments with error {error_id}: {error_ids.sum()}")

    unknown_error_ids = ~error_rows["error"].str.contains(
        "|".join(re.escape(error) for error in expected_errors.values()), regex=True
    )
    error_rows[unknown_error_ids].to_csv("error_unknown.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
