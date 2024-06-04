"""Run gather data as a slurm job with many cpus.

Example command:
python -m carps.analysis.gather_data_slurm 'rundir=runs/DEHB,runs/SMAC3-Hyperband,runs/SMAC3-MultiFidelityFacade' -m

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra

from carps.analysis.gather_data import filelogs_to_df

if TYPE_CHECKING:
    import pandas as pd
    from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="gather_data_slurm.yaml", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    return filelogs_to_df(cfg.rundir)


if __name__ == "__main__":
    main()
