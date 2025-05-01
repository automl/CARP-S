"""Scrape results from result_dir and create corresponding database entries."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import fire
from ConfigSpace import Configuration
from hydra.utils import instantiate
from omegaconf import OmegaConf
from py_experimenter.experimenter import PyExperimenter, ResultProcessor

from carps.analysis.gather_data import get_run_dirs, load_cfg, load_log
from carps.experimenter.create_cluster_configs import create_config_hash, fill_database
from carps.loggers.database_logger import DatabaseLogger
from carps.utils.loggingutils import get_logger, setup_logging
from carps.utils.trials import StatusType
from carps.utils.types import TrialInfo, TrialValue

setup_logging()
logger = get_logger(__file__)

experiment_configuration_file_path = Path(__file__).parent / "py_experimenter.yaml"

database_credential_file_path = Path(__file__).parent / "credentials.yaml"
if database_credential_file_path is not None and not database_credential_file_path.exists():
    database_credential_file_path = None  # type: ignore[assignment]

experimenter = PyExperimenter(
    experiment_configuration_file_path=experiment_configuration_file_path,
    name="carps",
    database_credential_file_path=database_credential_file_path,
    log_level=logging.INFO,
    use_ssh_tunnel=OmegaConf.load(experiment_configuration_file_path).PY_EXPERIMENTER.Database.use_ssh_tunnel,
)


def str_to_int(s: str) -> None | int:
    """Convert string to int or None.

    Parameters
    ----------
    s : str
        Input string
    """
    if s == "None":
        return None
    return int(s)


def str_to_float(s: str) -> None | float:
    """Convert string to float or None.

    Parameters
    ----------
    s : str
        Input string
    """
    if s == "None":
        return None
    return float(s)


def none_if_str_none(s: str) -> None | str:
    """Converts a string "None" to Python None if applicable.

    Parameters
    ----------
    s : str
        Input string
    """
    return None if s == "None" else s


def get_value(column_names: list[str], cfg_hash: str, column: str) -> Any:
    """Get value from db with given config hash.

    Parameters
    ----------
    column_names : list[str]
        Columns to consider
    cfg_hash : str
        Config hash to check for
    column : str
        Column to read
    """
    existing_rows = experimenter.db_connector._get_existing_rows(column_names)
    for e in existing_rows:
        if e["config_hash"] == cfg_hash:
            return e[column]
    raise RuntimeError("Expected config in database")


def main(rundir: str | list[str]) -> None:
    """Scrape all experiment data from rundir and write to database.

    Parameters
    ----------
    rundir : str | list[str]
        Run directory to fetch data from.

    """
    if isinstance(rundir, str):
        rundir = [rundir]
    rundirs_list = rundir

    for rundir_path in rundirs_list:
        logger.info(f"Get rundirs from {rundir_path}...")
        rundirs = get_run_dirs(rundir_path)
        logger.info(f"Found {len(rundirs)} runs. Scraping data...")

        for run_dir in rundirs:
            # Load configs to DB
            cfg = load_cfg(run_dir)

            if not cfg:
                raise RuntimeError(f"Config not found at {run_dir}")

            fill_database(cfg, experimenter=experimenter)

            cfg_hash = create_config_hash(cfg=cfg)
            column_names = list(experimenter.db_connector.database_configuration.keyfields.keys())

            status = get_value([*column_names, "status"], cfg_hash, "status")
            experiment_id = get_value([*column_names, "ID"], cfg_hash, "ID")

            if status is not None and status != "created":
                logger.info("Not scraping because runs are already in the database")
                continue

            # Update status
            connection = experimenter.db_connector.connect()
            cursor = connection.cursor()
            query = """
                UPDATE results
                SET status = 'scraped'
                WHERE ID = %s
            """
            cursor.execute(query, (experiment_id))
            connection.commit()
            cursor.close()
            connection.close()

            result_processor = ResultProcessor(
                experimenter.config.database_configuration,
                experimenter.db_connector,
                experiment_id=experiment_id,
                logger=experimenter.logger,
            )

            db_logger = DatabaseLogger(result_processor=result_processor)
            log = load_log(run_dir)
            inc = load_log(run_dir, "trajectory_logs.jsonl")

            configspace = instantiate(cfg.task.input_space.configuration_space)

            logger.info("Writing to database...")

            for row in log.itertuples(index=True):
                config = Configuration(
                    configuration_space=configspace,
                    values={x: ast.literal_eval(row.trial_info__config)[i] for i, x in enumerate(configspace)},
                )

                # Type checks are performed here because values may be "None", which isn't interpreted as null by the DB
                info = TrialInfo(
                    config,
                    str_to_int(row.trial_info__instance),
                    str_to_int(row.trial_info__seed),
                    str_to_float(row.trial_info__budget),
                    str_to_float(row.trial_info__normalized_budget),
                    none_if_str_none(row.trial_info__name),
                    none_if_str_none(row.trial_info__checkpoint),
                )
                value = TrialValue(
                    row.trial_value__cost,
                    row.trial_value__time,
                    row.trial_value__virtual_time,
                    StatusType(row.trial_value__status),
                    row.trial_value__starttime,
                    row.trial_value__endtime,
                )
                db_logger.log_trial(row.n_trials, info, value, row.n_function_calls)

            for row in inc.itertuples(index=True):
                config = Configuration(
                    configuration_space=configspace,
                    values={x: ast.literal_eval(row.trial_info__config)[i] for i, x in enumerate(configspace)},
                )
                info = TrialInfo(
                    config,
                    str_to_int(row.trial_info__instance),
                    str_to_int(row.trial_info__seed),
                    str_to_float(row.trial_info__budget),
                    str_to_float(row.trial_info__normalized_budget),
                    none_if_str_none(row.trial_info__name),
                    none_if_str_none(row.trial_info__checkpoint),
                )
                value = TrialValue(
                    row.trial_value__cost,
                    row.trial_value__time,
                    row.trial_value__virtual_time,
                    StatusType(row.trial_value__status),
                    row.trial_value__starttime,
                    row.trial_value__endtime,
                )
                db_logger.log_incumbent(row.n_trials, (info, value))
    logger.info("Done scraping")


if __name__ == "__main__":
    fire.Fire(main)
