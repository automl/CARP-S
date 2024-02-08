import json
import logging
import sys
from configparser import ConfigParser

import hydra
import sshtunnel
from omegaconf import DictConfig, OmegaConf
from py_experimenter.experimenter import PyExperimenter


def execute(experiment_configuration_file_path: str,
            database_credential_file: str,
            cfg: DictConfig,
            cfg_dict: dict):
    experimenter = PyExperimenter(experiment_configuration_file_path=experiment_configuration_file_path,
                                  name='smacbenchmarking',
                                  database_credential_file_path=database_credential_file,
                                  log_level=logging.INFO)

    cfg_json = OmegaConf.to_container(cfg, resolve=True)

    rows = [{
        'config': json.dumps(cfg_json),
        'problem_id': cfg_dict["benchmark_id"],
        'optimizer_id': cfg_dict["optimizer_id"],
    }]

    experimenter.fill_table_with_rows(rows)


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

    experiment_configuration_file_path = 'smacbenchmarking/container/py_experimenter.cfg'
    database_credential_file = 'smacbenchmarking/container/credentials.cfg'

    # configure debug-level logger
    logger = logging.Logger('SMAC evaluation')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    parsed_experiment_configuration_file = ConfigParser()
    parsed_experiment_configuration_file.read_file(experiment_configuration_file_path)

    if parsed_experiment_configuration_file['provider'] == 'mysql':
        with open(database_credential_file, 'r') as file:
            configparser = ConfigParser()
            configparser.read_file(file)
            config = configparser['TUNNEL_CONFIG']
            ssh_address_or_host = config['ssh_address_or_host']
            ssh_private_key_password = config['ssh_private_key_password']

        with sshtunnel.SSHTunnelForwarder(ssh_address_or_host=(ssh_address_or_host, 22),
                                          ssh_private_key_password=ssh_private_key_password,
                                          remote_bind_address=('127.0.0.1', 3306),
                                          local_bind_address=('127.0.0.1', 3306),
                                          logger=logger,
                                          ) as tunnel:

            return execute(experiment_configuration_file_path, database_credential_file, cfg, cfg_dict)
    else:
        return execute(experiment_configuration_file_path, database_credential_file, cfg, cfg_dict)


if __name__ == "__main__":
    main()
