"""Index all task and optimizer configs."""

from __future__ import annotations

from pathlib import Path

import fire
import omegaconf
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from carps.utils.loggingutils import get_logger, setup_logging
from carps.utils.requirements import maybe_get_carps_compatible_package_location

setup_logging()
logger = get_logger(__file__)

config_folder = Path(__file__).parent.parent / "configs"
config_folder_task = config_folder / "task"
config_folder_optimizer = config_folder / "optimizer"


def index_configs(carps_pkg_name: str | None = None) -> None:
    """Index all task and optimizer configs.

    Create `index.csv` containing the config filename `config_fn` and the
    `task_id` or `optimizer_id` for all task and optimizer configs.

    Parameters
    ----------
    carps_pkg_name : str | None
        Name of a CARPS-compatible package to import. This can be useful if the
        custom package defines OmegaConf resolvers which need to be registered.
    """
    task_index_fn = config_folder_task / "index.csv"
    optimizer_index_fn = config_folder_optimizer / "index.csv"

    table_list = []
    paths = list(config_folder.glob("**/*.yaml"))
    extra_loc = maybe_get_carps_compatible_package_location(carps_pkg_name)
    if extra_loc is not None:
        logger.info(f"Also indexing configs from package {carps_pkg_name} at {extra_loc}.")
        paths.extend(list(extra_loc.glob("**/*.yaml")))

    for path in tqdm(paths, total=len(paths)):
        cfg = OmegaConf.load(path)
        if "task_id" in cfg and "optimizer_id" in cfg:
            raise ValueError(f"Config {path} has both task_id and optimizer_id.")
        if "task_id" in cfg:
            key = "task_id"
        elif "optimizer_id" in cfg:
            key = "optimizer_id"
        else:
            continue
        try:
            value = cfg.get(key)
        except omegaconf.errors.InterpolationKeyError:
            logger.info(
                f"Could not read {key} from {path}. Maybe the config uses OmegaConf resolvers. " "Skipping this config."
            )
            continue
        table_list.append(
            {
                "type": key.replace("_id", ""),
                "config_fn": str(path),
                key: value,
            }
        )
    df_all = pd.DataFrame(table_list)

    df_task = df_all[df_all["type"] == "task"].drop(columns=["type"])
    df_optimizer = df_all[df_all["type"] == "optimizer"].drop(columns=["type"])

    df_task.to_csv(task_index_fn, index=False)
    df_optimizer.to_csv(optimizer_index_fn, index=False)
    logger.info(f"Wrote task index to {task_index_fn}.")
    logger.info(f"Wrote optimizer index to {optimizer_index_fn}.")


if __name__ == "__main__":
    fire.Fire(index_configs)
