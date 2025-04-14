"""Create experiments for CARPS and store experiment config in database."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from py_experimenter.exceptions import DatabaseConnectionError
from py_experimenter.experimenter import PyExperimenter

from carps.utils.loggingutils import CustomEncoder

logger = logging.getLogger("create experiments")

experiment_identifiers = ["optimizer_id", "task_id", "seed", "benchmark_id", "n_trials", "time_budget"]


def create_config_hash(cfg: DictConfig) -> str:
    """Create a hash of the configuration based on the experiment identifiers.

    Args:
        cfg (DictConfig): Configuration object.

    Returns:
        str: Hash of the configuration.
    """
    cfg_json = OmegaConf.to_container(cfg, resolve=True)
    reduced_cfg = {k: v for k, v in cfg_json.items() if k in experiment_identifiers}
    cfg_str = json.dumps(reduced_cfg, cls=CustomEncoder)
    return hashlib.sha256(cfg_str.encode()).hexdigest()


def create_config_hash_from_full_cfg(cfg: DictConfig) -> str:
    """Create a hash of the full configuration.

    Args:
        cfg (DictConfig): Configuration object.

    Returns:
            str: Hash of the full configuration.
    """
    cfg_json = OmegaConf.to_container(cfg, resolve=True)

    # This value will always be unique so it
    # disables duplicate checking when adding entries to the database.
    # Py_experimenter will add a creation date so the information
    # is not lost.
    if "timestamp" in cfg_json:
        del cfg_json["timestamp"]

    # Compute hash to efficiently compare configs.
    # In MySQL json objects are reordered to improve performance.
    # This means that there is no guarantee of the json strings
    # to be equal.
    cfg_str = json.dumps(cfg_json, cls=CustomEncoder)
    return hashlib.sha256(cfg_str.encode()).hexdigest()


def get_experiment_definition(cfg: OmegaConf) -> dict:
    """Get experiment definition from config.

    Args:
        cfg (OmegaConf): Configuration object.

    Returns:
        dict: Experiment definition / row for the database.
    """
    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)

    cfg_str = json.dumps(cfg_dict, cls=CustomEncoder)
    cfg_hash = create_config_hash(cfg)

    return {
        "config": cfg_str,
        "config_hash": cfg_hash,
        "benchmark_id": cfg_dict["benchmark_id"],
        "subset_id": cfg_dict.get("subset_id", None),
        "task_type": cfg_dict.get("task_type", None),
        "task_id": cfg_dict["task_id"],
        "optimizer_id": cfg_dict["optimizer_id"],
        "optimizer_container_id": cfg_dict["optimizer_container_id"],
        "seed": cfg_dict["seed"],
        "n_trials": cfg_dict["task"]["optimization_resources"]["n_trials"],
        "time_budget": cfg_dict["task"]["optimization_resources"]["time_budget"],
    }


@hydra.main(config_path="../configs", config_name="base.yaml", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Store experiment config in database.

    Parameters
    ----------
    cfg : DictConfig
        Global configuration.

    """
    experiment_definition = get_experiment_definition(cfg)

    column_names = list(experimenter.db_connector.database_configuration.keyfields.keys())
    exists = False
    try:
        existing_rows = experimenter.db_connector._get_existing_rows(column_names)

        # Check if experiment exists
        # Compare by hash only
        for e in existing_rows:
            if e["config_hash"] == experiment_definition["config_hash"]:
                exists = True
                logger.info("Experiment not added to the database because config hash already exists!")
    except DatabaseConnectionError as e:
        if "1146" in e.args[0] or "no such table" in e.args[0]:
            logger.info("Database empty, will fill.:)")
        else:
            raise e

    if not exists:
        # experimenter.db_connector.start_ssh_tunnel()
        experimenter.fill_table_with_rows([experiment_definition])
    # experimenter.close_ssh()


if __name__ == "__main__":
    # TODO make experiment_configuration_file_path and database_credential_file_path a commandline arg
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

    main()
