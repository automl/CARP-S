from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

def fix_legacy_task_id(task_id: str) -> str:
    task_id = "bbob/" + task_id if task_id.startswith("noiseless") else task_id
    return task_id.replace("noiseless/", "").replace("bb/tab/", "blackbox/tabular/").replace("MO/tab/", "multiobjective/tabular/").replace("hpobench/mf/", "hpobench/multifidelity/")

def create_subset_configs(subset_fn_dev: str, subset_fn_test: str, scenario: str) -> None:
    config_target_path = Path("carps/configs/task/subselection") / scenario
    config_target_path.mkdir(exist_ok=True, parents=True)

    def write_subsets(subset_fn: str, identifier: str):
        subset = pd.read_csv(subset_fn)
        subset["task_id"] = subset["problem_id"].apply(fix_legacy_task_id)
        subset_size = len(subset)
        task_ids = subset["task_id"].to_list()

        index_fn = config_target_path.parent.parent / "index.csv"
        if not index_fn.is_file():
            raise ValueError(f"Could not find {index_fn}. ObjectiveFunction ids have not been indexed. Run `python -m carps.utils.index_configs`.")
        task_index = pd.read_csv(index_fn)
        print(task_index.head())
        print(task_ids)
        not_found = [pid for pid in task_ids if pid not in task_index["task_id"].to_list()]
        if not_found:
                raise ValueError(f"Could not find {not_found} in {index_fn}. ObjectiveFunction ids have not been indexed. Run `python -m carps.utils.index_configs`.")

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
            yaml_str = f"# @package _global_\ntask_type: {scenario}\nsubset_id: {identifier}\n" + yaml_str
            new_fn.write_text(yaml_str)

    write_subsets(subset_fn_dev, "dev")
    write_subsets(subset_fn_test, "test")


if __name__ == "__main__":
    # python subselection/create_subset_configs.py subselection/data/Carps/BBv2_True_train_20_task_ids.csv subselection/data/Carps/BBv2_True_test_20_task_ids.csv blackbox
    # python subselection/create_subset_configs.py subselection/data/MF/lognorm/subset_20.csv subselection/data/MF/lognorm/subset_complement_subset_20.csv multifidelity
    # python subselection/create_subset_configs.py subselection/data/MO/lognorm/subset_10.csv subselection/data/MO/lognorm/subset_complement_subset_10.csv multiobjective
    # python subselection/create_subset_configs.py subselection/data/MOMF/lognorm/subset_9.csv subselection/data/MOMF/lognorm/subset_complement_subset_9.csv momf
    fire.Fire(create_subset_configs)

