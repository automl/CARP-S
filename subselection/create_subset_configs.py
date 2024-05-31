from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import fire


def create_subset_configs(subset_fn: str, scenario: str) -> None:
    config_target_path = Path("carps/configs/problem/subselection") / scenario
    config_target_path.mkdir(exist_ok=True, parents=True)

    subset = pd.read_csv(subset_fn)
    subset_size = len(subset)
    problem_ids = subset["problem_id"].to_list()

    index_fn = config_target_path.parent.parent / "index.csv"
    if not index_fn.is_file():
        raise ValueError(f"Could not find {index_fn}. Problem ids have not been indexed. Run `python -m carps.utils.index_configs`.")
    problem_index = pd.read_csv(index_fn)
    ids = [np.where(problem_index["problem_id"]==pid)[0][0] for pid in problem_ids]
    config_fns = problem_index["config_fn"][ids].to_list()

    for config_fn in config_fns:
        cfg = OmegaConf.load(config_fn)
        new_name = f"subset_{cfg.problem_id}.yaml".replace("/", "_")
        new_problem_id = f"{scenario}/{subset_size}/{cfg.problem_id}"
        cfg.problem_id = new_problem_id
        new_fn = config_target_path / new_name
        yaml_str = OmegaConf.to_yaml(cfg)
        yaml_str = "# @package _global_\n" + yaml_str
        new_fn.write_text(yaml_str)


if __name__ == "__main__":
    # python subselection/create_subset_configs.py subselection/run-data/subset_40.csv blackbox
    fire.Fire(create_subset_configs)

