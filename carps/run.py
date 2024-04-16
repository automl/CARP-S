from __future__ import annotations

import hydra
from omegaconf import DictConfig

from carps.utils.requirements import check_requirements
from carps.utils.running import optimize


@hydra.main(config_path="configs", config_name="base.yaml", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Run optimizer on problem.

    Save trajectory and metadata to database.

    Parameters
    ----------
    cfg : DictConfig
        Global configuration.

    """
    check_requirements(cfg=cfg)
    optimize(cfg=cfg)

    return None


if __name__ == "__main__":
    main()
