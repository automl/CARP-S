"""Index all task and optimizer configs."""

from __future__ import annotations

from pathlib import Path

import fire
import pandas as pd
from omegaconf import OmegaConf

config_folder = Path(__file__).parent.parent / "configs"
config_folder_task = config_folder / "task"
config_folder_optimizer = config_folder / "optimizer"


def index_configs() -> None:
    """Index all task and optimizer configs.

    Create `index.csv` containing the config filename `config_fn` and the
    `task_id` or `optimizer_id` for all task and optimizer configs.
    """
    for key, path in zip(["task_id", "optimizer_id"], [config_folder_task, config_folder_optimizer], strict=False):
        paths = list(path.glob("**/*.yaml"))

        table_list = []
        for p in paths:
            cfg = OmegaConf.load(p)
            value = cfg.get(key)
            table_list.append(
                {
                    "config_fn": str(p),
                    key: value,
                }
            )
        table = pd.DataFrame(table_list)
        table.to_csv(path / "index.csv", index=False)


if __name__ == "__main__":
    fire.Fire(index_configs)
