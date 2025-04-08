"""Scrape results from result_dir and create corresponding database entries."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import fire
from ConfigSpace import Configuration
from hydra.utils import instantiate
from omegaconf import OmegaConf
from py_experimenter.experimenter import PyExperimenter, ResultProcessor

import carps.experimenter.create_cluster_configs
from carps.analysis.gather_data import get_run_dirs, load_cfg, load_log
from carps.loggers.database_logger import DatabaseLogger
from carps.utils.loggingutils import CustomEncoder, get_logger, setup_logging
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


def str_to_val(s: str, type: str) -> None | int | float | str:
    """Convert string to given type or None.

    Parameters
    ----------
    s : str
        Input string
    type : str
        Type to convert to
    """
    if s == "None":
        return None
    if type == "int":
        return int(s)
    if type == "float":
        return float(s)
    return s


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

    for dir in rundirs_list:
        logger.info(f"Get rundirs from {dir}...")
        rundirs = get_run_dirs(dir)
        logger.info(f"Found {len(rundirs)} runs. Scraping data...")

        for rundir in rundirs:
            # Load configs to DB
            cfg = load_cfg(rundir)
            carps.experimenter.create_cluster_configs.main(cfg)

            cfg_json = OmegaConf.to_container(cfg, resolve=True)

            if "timestamp" in cfg_json:
                del cfg_json["timestamp"]

            cfg_str = json.dumps(cfg_json, cls=CustomEncoder)
            cfg_hash = hashlib.sha256(cfg_str.encode()).hexdigest()
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
            log = load_log(rundir)
            inc = load_log(rundir, "trajectory_logs.jsonl")

            configspace = instantiate(cfg.task.input_space.configuration_space)

            logger.info("Writing to database...")

            for row in log.itertuples(index=True):
                config = Configuration(
                    configuration_space=configspace,
                    values={x: ast.literal_eval(row.trial_info__config)[i] for i, x in enumerate(configspace)},
                )
                info = TrialInfo(
                    config,
                    str_to_val(row.trial_info__instance, "int"),
                    str_to_val(row.trial_info__seed, "int"),
                    str_to_val(row.trial_info__budget, "float"),
                    str_to_val(row.trial_info__normalized_budget, "float"),
                    str_to_val(row.trial_info__name, "str"),
                    str_to_val(row.trial_info__checkpoint, "str"),
                )
                value = TrialValue(
                    str_to_val(row.trial_value__cost, "str"),
                    str_to_val(row.trial_value__time, "float"),
                    str_to_val(row.trial_value__virtual_time, "float"),
                    StatusType(row.trial_value__status),
                    str_to_val(row.trial_value__starttime, "float"),
                    str_to_val(row.trial_value__endtime, "float"),
                )
                db_logger.log_trial(row.n_trials, info, value, row.n_function_calls)

            for row in inc.itertuples(index=True):
                config = Configuration(
                    configuration_space=configspace,
                    values={x: ast.literal_eval(row.trial_info__config)[i] for i, x in enumerate(configspace)},
                )
                info = TrialInfo(
                    config,
                    str_to_val(row.trial_info__instance, "int"),
                    str_to_val(row.trial_info__seed, "int"),
                    str_to_val(row.trial_info__budget, "float"),
                    str_to_val(row.trial_info__normalized_budget, "float"),
                    str_to_val(row.trial_info__name, "str"),
                    str_to_val(row.trial_info__checkpoint, "str"),
                )
                value = TrialValue(
                    str_to_val(row.trial_value__cost, "str"),
                    str_to_val(row.trial_value__time, "float"),
                    str_to_val(row.trial_value__virtual_time, "float"),
                    StatusType(row.trial_value__status),
                    str_to_val(row.trial_value__starttime, "float"),
                    str_to_val(row.trial_value__endtime, "float"),
                )
                db_logger.log_incumbent(row.n_trials, (info, value))
    logger.info("Done scraping")


if __name__ == "__main__":
    fire.Fire(main)
