"""Reset experiments that have errored out in the database."""

from __future__ import annotations

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

    yahpo_error_ids = error_rows["error"].str.contains("AttributeError: 'NoneType' object has no attribute 'update'")
    error_rows_yahpo = error_rows[yahpo_error_ids]
    error_rows_yahpo.to_csv("error_rows_yahpo.csv", index=False)  # type: ignore[attr-defined]
    error_rows_nonyahpo = error_rows[~yahpo_error_ids]
    error_rows_nonyahpo.to_csv("error_rows_nonyahpo.csv", index=False)  # type: ignore[attr-defined]


if __name__ == "__main__":
    fire.Fire(main)
