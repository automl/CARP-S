from __future__ import annotations
import fire
from pathlib import Path
from omegaconf import OmegaConf
import os
import pandas as pd
from carps.utils.loggingutils import get_logger

logger = get_logger(__file__)

config_folder = Path(__file__).parent.parent /  "configs"
config_folder_problem = config_folder / "problem"
config_folder_optimizer = config_folder / "optimizer" 

def find_override(problem_id: str | None = None, optimizer_id: str | None = None):
    if problem_id is not None:
        key = "problem_id"
        path = config_folder_problem
        to_find = problem_id
    elif optimizer_id is not None:
        key = "optimizer_id"
        path = config_folder_optimizer
        to_find = optimizer_id
    else:
        raise ValueError("Please specify either `problem_id` or `optimizer_id`.")
    
    index_fn = path / "index.csv"
    if not index_fn.is_file():
        cache_config_info()
    table = pd.read_csv(index_fn)

    try:
        config_fn = table["config_fn"][table[key]==to_find].values[0]
        stripped_path = str(config_fn)[len(str(config_folder))+1:-len(".yaml")]
        index = [x for x, v in enumerate(stripped_path) if v == '/'][0]
        override = "+" + stripped_path[:index] + "=" + stripped_path[index+1:]
        return override
    except Exception as e: 
        logger.info(f"Nothing found for {to_find} in config path {path}. Error: {e}")
        return None

def merge_overrides(overrides: list[str | None]) -> str:
    # overrides = [
    #     "+problem=YAHPO/MO/cfg_rbv2_super_1457",
    #     "+problem=YAHPO/MO/cfg_rbv2_super_1452",
    #     "+problem=YAHPO/MO/cfg_rbv2_super_1453",
    #     "+problem=YAHPO/MO/cfg_rbv2_super_1454",
    # ]
    overrides = [o for o in overrides if o is not None]
    if len(overrides) > 1:
        overrides = list(map(str, overrides))
        common = os.path.commonpath(overrides)
        index = common.find("=")
        override = common[:index+1] + ",".join([p[index+1:] for p in overrides])
    else:
        override = overrides[0]
    return override    

def cache_config_info():
    for key, path in zip(["problem_id", "optimizer_id"], [config_folder_problem, config_folder_optimizer]):
        paths = list(path.glob("**/*.yaml"))

        table = []
        for p in paths:
            cfg = OmegaConf.load(p)
            value = cfg.get(key)
            table.append({
                "config_fn": str(p),
                key: value,
            })
        table = pd.DataFrame(table)
        table.to_csv(path / "index.csv", index=False)


if __name__ == "__main__":
    fire.Fire(find_override)
    # cache_config_info()
