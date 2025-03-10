from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def create_subset_configs(subset_fn_dev: str, subset_fn_test: str, scenario: str) -> None:
    config_target_path = Path("carps/configs/task/subselection") / scenario
    config_target_path.mkdir(exist_ok=True, parents=True)

    def write_subsets(subset_fn: str, identifier: str):
        subset = pd.read_csv(subset_fn)
        subset["task_id"] = subset["task_id"].apply(lambda x: "bbob/" + x if x.startswith("noiseless") else x)
        subset_size = len(subset)
        task_ids = subset["task_id"].to_list()

        index_fn = config_target_path.parent.parent / "index.csv"
        if not index_fn.is_file():
            raise ValueError(f"Could not find {index_fn}. ObjectiveFunction ids have not been indexed. Run `python -m carps.utils.index_configs`.")
        task_index = pd.read_csv(index_fn)
        ids = [np.where(task_index["task_id"]==pid)[0][0] for pid in task_ids]
        config_fns = task_index["config_fn"][ids].to_list()

        for config_fn in config_fns:
            cfg = OmegaConf.load(config_fn)
            new_name = f"subset_{cfg.task_id}.yaml".replace("/", "_")
            new_task_id = f"{scenario}/{subset_size}/{identifier}/{cfg.task_id}"
            cfg.task_id = new_task_id
            new_fn = config_target_path / identifier / new_name
            new_fn.parent.mkdir(exist_ok=True, parents=True)
            yaml_str = OmegaConf.to_yaml(cfg)
            yaml_str = f"# @package _global_\nscenario: {scenario}\nsubset_id: {identifier}\n" + yaml_str
            new_fn.write_text(yaml_str)

    write_subsets(subset_fn_dev, "dev")
    write_subsets(subset_fn_test, "test")


if __name__ == "__main__":
    # python subselection/create_subset_configs.py subselection/BB_2/default/subset_30.csv subselection/BB_2/default/subset_complement_subset_30.csv blackbox
    # python subselection/create_subset_configs.py subselection/MF_1/lognorm/subset_20.csv subselection/MF_1/lognorm/subset_complement_subset_20.csv multifidelity
    # python subselection/create_subset_configs.py subselection/MO_0/lognorm/subset_10.csv subselection/MO_0/lognorm/subset_complement_subset_10.csv multiobjective
    # python subselection/create_subset_configs.py subselection/MOMF_0/lognorm/subset_9.csv subselection/MOMF_0/lognorm/subset_complement_subset_9.csv momf
    fire.Fire(create_subset_configs)

