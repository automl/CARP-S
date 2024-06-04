from __future__ import annotations

from pathlib import Path

import fire
import pandas as pd
from omegaconf import OmegaConf

config_folder = Path(__file__).parent.parent / "configs"
config_folder_problem = config_folder / "problem"
config_folder_optimizer = config_folder / "optimizer"


def index_configs():
    """Index all problem and optimizer configs.

    Create `index.csv` containing the config filename `config_fn` and the
    """
    for key, path in zip(
        ["problem_id", "optimizer_id"], [config_folder_problem, config_folder_optimizer], strict=False
    ):
        paths = list(path.glob("**/*.yaml"))

        table = []
        for p in paths:
            cfg = OmegaConf.load(p)
            value = cfg.get(key)
            table.append(
                {
                    "config_fn": str(p),
                    key: value,
                }
            )
        table = pd.DataFrame(table)
        table.to_csv(path / "index.csv", index=False)


if __name__ == "__main__":
    fire.Fire(index_configs)
