import hydra
import sys
from omegaconf import DictConfig
import smac

from smac import Scenario

import os
from hydra.core.hydra_config import HydraConfig


from typing import Any

from ConfigSpace import Configuration, ConfigurationSpace, Float
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import print as printr
from rich import inspect
from functools import partial
import numpy as np
from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.optimizers.optimizer import Optimizer
from hydra.utils import instantiate, get_class
from smacbenchmarking.utils.exceptions import NotSupportedError
import pandas as pd
from pathlib import Path
import json


def make_problem(cfg: DictConfig) -> Problem:
    problem_cfg = cfg.problem
    problem = instantiate(problem_cfg)
    return problem


def make_optimizer(cfg: DictConfig, problem: Problem) -> Optimizer:
    optimizer_cfg = cfg.optimizer
    optimizer = instantiate(optimizer_cfg)(problem=problem)
    return optimizer


def save_run(cfg: DictConfig, optimizer: Optimizer, metadata: dict):
    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
    # cfg_dict = pd.json_normalize(cfg_dict, sep=".").iloc[0].to_dict()  # flatten cfg

    trajectory_data = {}
    for sort_by in ["trials", "walltime"]:
        try:
            X, Y = optimizer.get_trajectory(sort_by=sort_by)
            trajectory_data[sort_by] = {"X": X, "Y": Y}
        except NotSupportedError:
            continue

    data = {
        "cfg": cfg_dict,
        "metadata": metadata,
        "rundata": {
            "trajectory": trajectory_data,
        },
    }

    filename = "rundata.json"
    with open(filename, "w") as file:
        json.dump(data, file, indent="\t")


@hydra.main(config_path="configs", config_name="base.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
    printr(cfg_dict)
    hydra_cfg = HydraConfig.instance().get()
    printr(hydra_cfg.run.dir)

    problem = make_problem(cfg=cfg)
    inspect(problem)

    optimizer = make_optimizer(cfg=cfg, problem=problem)
    inspect(optimizer)

    try:
        optimizer.run()
    except NotSupportedError:
        print("Not supported. Skipping.")
    except Exception as e:
        print("Something went wrong:")
        print(e)

    metadata = {"hi": "hello"}  # TODO add reasonable meta data

    save_run(cfg=cfg, optimizer=optimizer, metadata=metadata)

    return None


if __name__ == "__main__":
    main()
