"""Index all task and optimizer configs."""

from __future__ import annotations

from pathlib import Path

import fire
import pandas as pd
from omegaconf import OmegaConf
from rich.progress import track

from carps.utils.loggingutils import get_logger

logger = get_logger("ConfigIndexer")

config_folder = Path(__file__).parent.parent / "configs"
config_folder_task = config_folder / "task"
config_folder_optimizer = config_folder / "optimizer"


def index_configs(extra_task_paths: list[str] | None = None, extra_optimizer_paths: list[str] | None = None) -> None:
    """Index all task and optimizer configs.

    Create `index.csv` containing the config filename `config_fn` and the
    `task_id` or `optimizer_id` for all task and optimizer configs.

    Parameters
    ----------
    extra_task_paths : list[str], optional
        Extra paths to custom tasks, must be a folder containing only task configs.
    extra_optimizer_paths : list[str], optional
        Extra paths to custom optimizers, must be a folder containing only optimizer configs.
    """
    config_folder_tasks = [config_folder_task] if extra_task_paths is None else [config_folder_task, *extra_task_paths]  # type: ignore[list-item]
    config_folder_tasks = [Path(p) for p in config_folder_tasks]
    config_folder_optimizers = (
        [config_folder_optimizer]
        if extra_optimizer_paths is None
        else [config_folder_optimizer, *extra_optimizer_paths]  # type: ignore[list-item]
    )
    config_folder_optimizers = [Path(p) for p in config_folder_optimizers]
    for key, paths in zip(["task_id", "optimizer_id"], [config_folder_tasks, config_folder_optimizers], strict=False):
        logger.info(f"Search configs for {key} from {paths}...")
        filenames = []
        for path in paths:
            filenames.extend(list(path.glob("**/*.yaml")))

        table_list = []
        for fn in track(filenames, total=len(filenames), description=f"Gathering for {key}..."):
            cfg = OmegaConf.load(fn)
            value = cfg.get(key)
            table_list.append(
                {
                    "config_fn": str(fn),
                    key: value,
                }
            )
        table = pd.DataFrame(table_list)
        table.to_csv(paths[0] / "index.csv", index=False)


if __name__ == "__main__":
    fire.Fire(index_configs)
