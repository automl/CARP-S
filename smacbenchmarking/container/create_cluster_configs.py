import json
import logging
import os
import sys
from configparser import ConfigParser

import hydra
from omegaconf import DictConfig, OmegaConf
from py_experimenter.experimenter import PyExperimenter


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
                                  use_ssh_tunnel=True,)

    cfg_json = OmegaConf.to_container(cfg, resolve=True)

    rows = [{
        "config": json.dumps(cfg_json),
        "benchmark_id": cfg_dict["benchmark_id"],
        "problem_id": cfg_dict["benchmark_id"],
        "optimizer_id": cfg_dict["optimizer_id"],
        "optimizer_container_id": cfg_dict["optimizer_container_id"],
        "seed": cfg_dict["seed"],
        "task__n_trials": cfg_dict["task"]["n_trials"],
    }]

    experimenter.fill_table_with_rows(rows)

    # TODO: Return array info
    return None


if __name__ == "__main__":
    main()
