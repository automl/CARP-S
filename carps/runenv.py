from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.core.hydra_config import HydraConfig

from carps.utils.loggingutils import get_logger, setup_logging

if TYPE_CHECKING:
    from omegaconf import DictConfig

setup_logging()
logger = get_logger(__file__)


@hydra.main(config_path="configs", config_name="base.yaml", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Run optimizer on problem.

    Save trajectory and metadata to database.

    Parameters
    ----------
    cfg : DictConfig
        Global configuration.

    """
    hydra_cfg = HydraConfig.instance().get()
    overrides = hydra_cfg.overrides.task
    runcommand = f"python -m carps.run {' '.join(overrides)}"
    logger.info(f"Runcommand: `{runcommand}`")
    env_location = Path(cfg.conda_env_location) / cfg.conda_env_name
    logger.info(f"Selected environment: {env_location}")
    command = f"source ~/.bashrc; micromamba run -p {env_location} {runcommand}"
    logger.info(command)
    subprocess.run(command, shell=True, check=False)


if __name__ == "__main__":
    main()
