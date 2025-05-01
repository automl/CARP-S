from __future__ import annotations

from core import select_sets_sequential, select_sets_split

import hydra
from omegaconf import OmegaConf

from carps.utils.loggingutils import get_logger, setup_logging
import pandas as pd
from pathlib import Path
import numpy as np
import os
import fire

setup_logging()
logger = get_logger("Subselect")


def _subselect(
    fullset_csv_path: str | Path,
    subset_size: int = 10,
    n_reps: int = 5000,
    method: str = "sequential",
    log_transform: bool = False,  # noqa: FBT001, FBT002
    executable: str = "./a.out",
    subset_ids: tuple[str] = ("dev", "test"),
    output_subset_file: str = "subsets.csv",
    output_metadata_file: str = "metadata.csv",
):
    print(os.getcwd())

    # Load points DataFrame
    points_df = pd.read_csv(fullset_csv_path)

    # Fix index
    points_df = points_df.rename(columns={"problem_id": "task_id"})
    points_df = points_df.set_index("task_id")
    points_df.index.name = "task_id"
    print(points_df.head())

    # assert all(0<= points_df.to_numpy() <= 1), "Values in the DataFrame must be in the unit cube [0, 1]."
    # Min max scale to unit cube
    points_df = points_df.sub(points_df.min(axis=1), axis=0)
    points_df = points_df.div(points_df.max(axis=1), axis=0)

    if log_transform:
        # Apply log transformation to the DataFrame
        points_df = points_df.map(lambda x: np.log10(x + 1e-10))
        # Stretch to unit cube per row
        points_df = points_df.sub(points_df.min(axis=1), axis=0)
        points_df = points_df.div(points_df.max(axis=1), axis=0)
        # points_df = points_df.fillna(0)


    # Select sets using the specified method
    if method == "sequential":
        subset_df, metadata = select_sets_sequential(
            points_df, subset_size, n_reps, executable, subset_ids)
    elif method == "split":
        subset_df, metadata = select_sets_split(
            points_df, subset_size, n_reps, executable, subset_ids)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Save the selected subsets and metadata
    subset_df.to_csv(output_subset_file, index=True)
    metadata.to_csv(output_metadata_file, index=False)
    return subset_df, metadata

@hydra.main(config_path=".", config_name="subselect", version_base=None)
def main(cfg: OmegaConf):
    """Main function to run the subsetting process.

    Args:
        cfg (OmegaConf): Configuration object containing parameters for subsetting.
    """
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")
    return _subselect(
        fullset_csv_path=cfg.fullset_csv_path,
        subset_size=cfg.subset_size,
        n_reps=cfg.n_reps,
        method=cfg.method,
        log_transform=cfg.log_transform,
        output_subset_file=cfg.output_subset_file,
        output_metadata_file=cfg.output_metadata_file,
        executable=cfg.executable,
        subset_ids=tuple(cfg.subset_ids),
    )

if __name__ == "__main__":
    main()