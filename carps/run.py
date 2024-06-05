from __future__ import annotations

from typing import TYPE_CHECKING

import hydra
from hydra.core.hydra_config import HydraConfig

from carps.utils.loggingutils import get_logger, setup_logging
from carps.utils.requirements import check_requirements
from carps.utils.running import optimize

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
    logger.info(f"Runcommand: `python -m carps.run {' '.join(overrides)}`")
    check_requirements(cfg=cfg)
    optimize(cfg=cfg)


if __name__ == "__main__":
    main()
