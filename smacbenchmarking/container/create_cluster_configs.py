import json
import logging
import os
import sys
from configparser import ConfigParser
import hashlib
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.exceptions import DatabaseConnectionError

logger = logging.getLogger("create experiments")


@hydra.main(config_path="../configs", config_name="base.yaml", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Run optimizer on problem.

    Save trajectory and metadata to database.

    Parameters
    ----------
    cfg : DictConfig
        Global configuration.

    """
    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)

    experiment_configuration_file_path = "smacbenchmarking/container/py_experimenter.yaml"

    if os.path.exists("smacbenchmarking/container/credentials.yaml"):
        database_credential_file = "smacbenchmarking/container/credentials.yaml"
    else:
        database_credential_file = None

    experimenter = PyExperimenter(experiment_configuration_file_path=experiment_configuration_file_path,
                                  name="smacbenchmarking",
                                  database_credential_file_path=database_credential_file,
                                  log_level=logging.INFO,
                                  use_ssh_tunnel=OmegaConf.load(experiment_configuration_file_path).PY_EXPERIMENTER.Database.use_ssh_tunnel
                                  )

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
    cfg_str = json.dumps(cfg_json)
    cfg_hash = hashlib.sha256(cfg_str.encode()).hexdigest()

    rows = [{
        "config": cfg_str,
        "config_hash": cfg_hash,
        "benchmark_id": cfg_dict["benchmark_id"],
        "problem_id": cfg_dict["problem_id"],
        "optimizer_id": cfg_dict["optimizer_id"],
        "optimizer_container_id": cfg_dict["optimizer_container_id"],
        "seed": cfg_dict["seed"],
        "n_trials": cfg_dict["task"]["n_trials"],
        "time_budget": cfg_dict["task"]["time_budget"],
    }]

    column_names = list(experimenter.db_connector.database_configuration.keyfields.keys())

    exists = False

    try:
        existing_rows = experimenter.db_connector._get_existing_rows(column_names)

        # Check if experiment exists
        # Compare by hash only
        for e in existing_rows:
            if e["config_hash"] == cfg_hash:
                exists = True
                logger.info("Experiment not added to the database because config hash already exists!")
    except DatabaseConnectionError as e:
        if "1146" in e.args[0]:
            logger.info("Database empty, will fill.:)")
        else:
            raise e

    if not exists:
        experimenter.fill_table_with_rows(rows)

    return None


if __name__ == "__main__":
    main()
