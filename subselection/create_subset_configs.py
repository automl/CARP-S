from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import fire
import shutil


def create_subset_configs(subset_fn_dev: str, subset_fn_test: str, scenario: str) -> None:
    config_target_path = Path("carps/configs/problem/subselection") / scenario
    config_target_path.mkdir(exist_ok=True, parents=True)
    shutil.rmtree(config_target_path)

    def write_subsets(subset_fn: str, identifier: str):
        subset = pd.read_csv(subset_fn)
        subset["problem_id"] = subset["problem_id"].apply(lambda x: "bbob/" + x if x.startswith("noiseless") else x)
        subset_size = len(subset)
        problem_ids = subset["problem_id"].to_list()

        index_fn = config_target_path.parent.parent / "index.csv"
        if not index_fn.is_file():
            raise ValueError(f"Could not find {index_fn}. Problem ids have not been indexed. Run `python -m carps.utils.index_configs`.")
        problem_index = pd.read_csv(index_fn)
        try:
            ids = [np.where(problem_index["problem_id"]==pid)[0][0] for pid in problem_ids]
        except IndexError:
            raise ValueError(f"Could not find all problem ids in {index_fn}.  Run `python -m carps.utils.index_configs`. "\
                             "If that does not help, check the subset file.")
        config_fns = problem_index["config_fn"][ids].to_list()

        for config_fn in config_fns:
            cfg = OmegaConf.load(config_fn)
            new_name = f"subset_{cfg.problem_id}.yaml".replace("/", "_")
            new_problem_id = f"{scenario}/{subset_size}/{identifier}/{cfg.problem_id}"
            cfg.problem_id = new_problem_id
            new_fn = config_target_path / identifier / new_name
            new_fn.parent.mkdir(exist_ok=True, parents=True)
            yaml_str = OmegaConf.to_yaml(cfg)
            yaml_str = "# @package _global_\n" + yaml_str
            new_fn.write_text(yaml_str)

    write_subsets(subset_fn_dev, "dev")
    write_subsets(subset_fn_test, "test")


if __name__ == "__main__":
    # python subselection/create_subset_configs.py subselection/BB_2/default/subset_30.csv subselection/BB_2/default/subset_complement_subset_30.csv blackbox
    # python subselection/create_subset_configs.py subselection/MF_1/lognorm/subset_20.csv subselection/MF_1/lognorm/subset_complement_subset_20.csv multifidelity
    # python subselection/create_subset_configs.py subselection/MO_0/lognorm/subset_10.csv subselection/MO_0/lognorm/subset_complement_subset_10.csv multiobjective
    # python subselection/create_subset_configs.py subselection/MOMF_0/lognorm/subset_9.csv subselection/MOMF_0/lognorm/subset_complement_subset_9.csv momf
    
    
    # python subselection/create_subset_configs.py subselection/data/2024_11/BBv2_True_dev_20.csv subselection/data/2024_11/BBv2_True_test_20.csv blackbox
    # python subselection/create_subset_configs.py subselection/data/2024_11/MOv2_True_dev_10.csv subselection/data/2024_11/MOv2_True_test_10.csv multiobjective
    fire.Fire(create_subset_configs)

