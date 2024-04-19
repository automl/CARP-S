from __future__ import annotations

import json
import logging
import multiprocessing
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fire
import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from hydra.core.utils import setup_globals
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from smac.runhistory.dataclasses import TrialInfo

if TYPE_CHECKING:
    from carps.benchmarks.problem import Problem

FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])


setup_globals()

def get_run_dirs(outdir: str):
    triallog_files = list(Path(outdir).glob("**/trial_logs.jsonl"))
    return [f.parent for f in triallog_files]

def annotate_with_cfg(df: pd.DataFrame, cfg: DictConfig, config_keys: list[str], config_keys_forbidden: list[str] | None = None) -> pd.DataFrame:
    if config_keys_forbidden is None:
        config_keys_forbidden = []
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    flat_cfg = pd.json_normalize(cfg_resolved, sep=".").iloc[0].to_dict()
    for k, v in flat_cfg.items():
        if np.any([k.startswith(c) for c in config_keys]) and not np.any([c in k for c in config_keys_forbidden]):
            df[k] = v
    return df

def get_Y(X: np.ndarray, problem: Problem) -> np.ndarray:
    return np.array(
        [
            problem.evaluate(
                TrialInfo(
                    config=Configuration(
                        configuration_space=problem.configspace,
                        vector=x,
                    )
                )
            ).cost
            for x in X
        ]
    )

def join_df(df1: pd.DataFrame, df2: pd.DataFrame, on: str = "n_trials") -> pd.DataFrame:
    df1.set_index(on)
    return df1.join(df2.set_index(on), on=on)

def load_log(
    rundir: Path
) -> pd.DataFrame:
    df = read_trial_log(rundir)

    cfg = load_cfg(rundir)
    config_fn = str(Path(rundir) / ".hydra/config.yaml")
    cfg_str = OmegaConf.to_yaml(cfg=cfg)
    df["cfg_fn"] = config_fn
    df["cfg_str"] = [(config_fn, cfg_str)] * len(df)

    # df = maybe_add_bandit_log(df, rundir, n_initial_design=cfg.task.n_initial_design)

    config_keys = ["benchmark", "problem", "seed", "optimizer_id"]
    config_keys_forbidden = ["_target_", "_partial_"]
    df = annotate_with_cfg(df=df, cfg=cfg, config_keys=config_keys, config_keys_forbidden=config_keys_forbidden)
    if "problem.function.seed" in df:
        df = df.drop(columns=["problem.function.seed"])
    if "problem.function.dim" in df:
        df = df.rename(columns={"problem.function.dim": "dim"})
    return df

def map_multiprocessing(
    task_function: Callable,
    task_params: list[Any],
    n_processes: int = 4,
) -> list:
    with multiprocessing.Pool(processes=n_processes) as pool:
        return pool.map(task_function, task_params)

def read_jsonl_content(filename: str) -> pd.DataFrame:
    with open(filename) as file:
        content = [json.loads(l) for l in file.readlines()]
    return pd.DataFrame(content)

def read_trial_log(rundir: str):
    path = Path(rundir) / "trial_logs.jsonl"
    df = None
    if path.exists():
        df = read_jsonl_content(path)
        df = normalize_drop(df, "trial_info", rename_columns=True, sep="__")
        df = normalize_drop(df, "trial_value", rename_columns=True, sep="__")
        # df = df.drop(columns=["trial_info__instance", "trial_info__budget", "trial_value__time", "trial_value__status", "trial_value__starttime", "trial_value__endtime"])
    return df

def load_cfg(rundir: str) -> DictConfig:
    config_fn = Path(rundir) / ".hydra/config.yaml"
    return OmegaConf.load(config_fn)

def normalize_drop(df: pd.DataFrame, key: str, rename_columns: bool = False, sep: str = ".") -> pd.DataFrame:
    """Normalize columns containing dicts.

    Parameters
    ----------
    df : pd.DataFrame
        Source data frame
    key : str
        Column name containing dicts
    rename_columns : bool, default False
        If true, rename columns to key.dict_key, if false, use dict keys as column names.
    sep : str, default .
        If rename columns: pattern for column is <key><sep><value>

    Returns:
    --------
    pd.DataFrame
        Flat data frame
    """
    df_tmp = pd.DataFrame(df[key].tolist())
    if "additional_info" in df_tmp and df_tmp["additional_info"].iloc[0] == {}:
        del df_tmp["additional_info"]
    if rename_columns:
        df_tmp = df_tmp.rename(columns={c: f"{key}{sep}{c}" for c in df_tmp.columns})
    return pd.concat([df.drop(key, axis=1), df_tmp], axis=1)

def maybe_add_n_trials(df: pd.DataFrame, n_initial_design: int, counter_key: str = "n_calls") -> pd.DataFrame:
    if "n_trials" not in df:
        df["n_trials"] = df[counter_key] + n_initial_design  # n_trials is 1-based
    return df

def filelogs_to_df(rundir: str) -> None:
    rundirs = get_run_dirs(rundir)
    results = map_multiprocessing(load_log, rundirs, n_processes=4)
    df = pd.concat(results).reset_index(drop=True)
    df_cfg = pd.DataFrame([{"cfg_fn": k, "cfg_str": v}  for k, v in df["cfg_str"].unique()])
    df_cfg.loc[:, "experiment_id"] = np.arange(0, len(df_cfg))
    df_cfg.loc[:, "cfg_str"] = df_cfg["cfg_str"].apply(lambda x: x.replace("\n", "\\n"))
    df["experiment_id"] = df["cfg_fn"].apply(lambda x:  np.where(df_cfg["cfg_fn"].to_numpy()==x)[0][0])
    del df["cfg_str"]
    del df["cfg_fn"]
    df.to_csv(Path(rundir) / "logs.csv", index=False)
    df_cfg.to_csv(Path(rundir) / "logs_cfg.csv", index=False)
    return None


if __name__ == "__main__":
    fire.Fire(filelogs_to_df)
