from __future__ import annotations
import fire
from pathlib import Path
from omegaconf import OmegaConf
import os
from carps.utils.loggingutils import setup_logging, get_logger

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

    paths = list(path.glob("**/*.yaml"))
    for p in paths:
        cfg = OmegaConf.load(p)
        value = cfg.get(key)
        if value == to_find:
            stripped_path = str(p)[len(str(config_folder))+1:-len(".yaml")]
            index = [x for x, v in enumerate(stripped_path) if v == '/'][0]
            override = "+" + stripped_path[:index] + "=" + stripped_path[index+1:]
            return override
    logger.info(f"Nothing found for {to_find} in config path {path}")
    return None

def merge_overrides(overrides: list[str]) -> str:
    # overrides = [
    #     "+problem=YAHPO/MO/cfg_rbv2_super_1457",
    #     "+problem=YAHPO/MO/cfg_rbv2_super_1452",
    #     "+problem=YAHPO/MO/cfg_rbv2_super_1453",
    #     "+problem=YAHPO/MO/cfg_rbv2_super_1454",
    # ]
    if len(overrides) > 1:
        overrides = map(str, overrides)
        common = os.path.commonpath(overrides)
        index = common.find("=")
        override = common[:index+1] + ",".join([p[index+1:] for p in overrides])
    else:
        override = overrides[0]
    return override    

if __name__ == "__main__":
    fire.Fire(find_override)
