"""Reset experiments that have errored out in the database."""

from __future__ import annotations

from pathlib import Path

import fire
from omegaconf import OmegaConf
from py_experimenter.experimenter import PyExperimenter

from carps.utils.loggingutils import get_logger, setup_logging

setup_logging()
logger = get_logger(__file__)


def main(
    pyexperimenter_configuration_file_path: str | None = None,
    database_credential_file_path: str | Path | None = None,
    outdir: str | Path | None = None,
) -> None:
    """Download results from the database and save them to outdir.

    Args:
        pyexperimenter_configuration_file_path (str, optional): Path to the py_experimenter configuration file.
            Defaults to None.
        database_credential_file_path (str | Path, optional): Path to the database credential file. Defaults to None.
        outdir (str | Path, optional): Directory to save the results. Defaults to None, will then be
            'experimenter/results'.
    """
    outdir = outdir or Path("experimenter") / "results"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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
        name="download_results",
        database_credential_file_path=database_credential_file_path,
        log_file="logs/reset_experiments.log",
        use_ssh_tunnel=OmegaConf.load(experiment_configuration_file_path).PY_EXPERIMENTER.Database.use_ssh_tunnel,
    )

    experiment_config_table = experimenter.get_table()
    trajectory_table = experimenter.get_logtable("trajectory")
    trials_table = experimenter.get_logtable("trials")
    codecarbon_table = experimenter.get_codecarbon_table()

    experiment_config_table.to_parquet(outdir / "experiment_config.parquet", index=False)
    trajectory_table.to_parquet(outdir / "trajectory.parquet", index=False)
    trials_table.to_parquet(outdir / "trials.parquet", index=False)
    codecarbon_table.to_parquet(outdir / "codecarbon.parquet", index=False)
    logger.info(
        "Downloaded results from the database. "
        f"Saved to '{outdir}'. "
        "You can process them into a single table for the plotting scripts with "
        "`python -m carps.experimenter.database.process_logs`."
    )


if __name__ == "__main__":
    fire.Fire(main)
