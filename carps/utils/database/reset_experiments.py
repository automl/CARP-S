from __future__ import annotations

from pathlib import Path

import fire
from omegaconf import OmegaConf
from py_experimenter.experiment_status import ExperimentStatus
from py_experimenter.experimenter import PyExperimenter


def main(
    pyexperimenter_configuration_file_path: str | None = None, database_credential_file_path: str | None = None
) -> None:
    experiment_configuration_file_path = (
        pyexperimenter_configuration_file_path or Path(__file__).parent.parent.parent / "container/py_experimenter.yaml"
    )

    database_credential_file_path = (
        database_credential_file_path or Path(__file__).parent.parent.parent / "container/credentials.yaml"
    )
    if database_credential_file_path is not None and not database_credential_file_path.exists():
        database_credential_file_path = None

    experimenter = PyExperimenter(
        experiment_configuration_file_path=experiment_configuration_file_path,
        name="remove_error_rows",
        database_credential_file_path=database_credential_file_path,
        log_file="logs/reset_experiments.log",
        use_ssh_tunnel=OmegaConf.load(experiment_configuration_file_path).PY_EXPERIMENTER.Database.use_ssh_tunnel,
    )
    experimenter.db_connector.reset_experiments(ExperimentStatus.ERROR.value)


if __name__ == "__main__":
    fire.Fire(main)
