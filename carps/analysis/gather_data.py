from __future__ import annotations

import json
import multiprocessing
from collections.abc import Callable, Iterable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import fire
import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from hydra.core.utils import setup_globals
from omegaconf import DictConfig, ListConfig, OmegaConf

from carps.utils.loggingutils import get_logger, setup_logging
from carps.utils.task import Task
from carps.utils.trials import TrialInfo

if TYPE_CHECKING:
    from carps.benchmarks.problem import Problem

setup_logging()
logger = get_logger(__file__)


setup_globals()


def glob_trial_logs(p: str) -> list[str]:
    return list(Path(p).glob("**/trial_logs.jsonl"))


def get_run_dirs(outdir: str):
    opt_paths = list(Path(outdir).glob("*/*"))
    with multiprocessing.Pool() as pool:
        triallog_files = pool.map(glob_trial_logs, opt_paths)
    triallog_files = np.concatenate(triallog_files)
    return [f.parent for f in triallog_files]


def annotate_with_cfg(
    df: pd.DataFrame,
    cfg: DictConfig,
    config_keys: list[str],
    config_keys_forbidden: list[str] | None = None,
) -> pd.DataFrame:
    if config_keys_forbidden is None:
        config_keys_forbidden = []
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    flat_cfg = pd.json_normalize(cfg_resolved, sep=".").iloc[0].to_dict()  # type: ignore
    for k, v in flat_cfg.items():
        if np.any([k.startswith(c) for c in config_keys]) and not np.any([c in k for c in config_keys_forbidden]):
            if isinstance(v, list | ListConfig):
                v = [v] * len(df)
            df[k] = v
    return df


def get_Y(X: np.ndarray, problem: Problem) -> np.ndarray:
    return np.array(
        [
            (
                problem.evaluate(
                    trial_info=TrialInfo(config=Configuration(configuration_space=problem.configspace, vector=x))
                ).cost
            )
            for x in X
        ]
    )


def join_df(df1: pd.DataFrame, df2: pd.DataFrame, on: str = "n_trials") -> pd.DataFrame:
    df1.set_index(on)
    return df1.join(df2.set_index(on), on=on)


def load_log(rundir: str | Path, log_fn: str = "trial_logs.jsonl") -> pd.DataFrame:
    df = read_trial_log(rundir, log_fn=log_fn)
    if df is None:
        # raise NotImplementedError("No idea what should happen here!?")
        return pd.DataFrame()

    cfg = load_cfg(rundir)
    if cfg is not None:
        config_fn = str(Path(rundir) / ".hydra/config.yaml")
        cfg_str = OmegaConf.to_yaml(cfg=cfg)
        df["cfg_fn"] = config_fn
        df["cfg_str"] = [(config_fn, cfg_str)] * len(df)

        config_keys = ["benchmark", "problem", "seed", "optimizer_id", "task"]
        config_keys_forbidden = ["_target_", "_partial_"]
        df = annotate_with_cfg(df=df, cfg=cfg, config_keys=config_keys, config_keys_forbidden=config_keys_forbidden)
        # df = maybe_add_bandit_log(df, rundir, n_initial_design=cfg.task.n_initial_design)
    else:
        config_fn = "no_hydra_config"
        cfg_str = ""
        df["cfg_fn"] = config_fn
        df["cfg_str"] = [(config_fn, cfg_str)] * len(df)

    if "problem.function.seed" in df:
        df = df.drop(columns=["problem.function.seed"])

    if "problem.function.dim" in df:
        df = df.rename(columns={"problem.function.dim": "dim"})

    return process_logs(df)


T = TypeVar("T")
R = TypeVar("R")


def map_multiprocessing(
    task_function: Callable[[T], R],
    task_params: Iterable[T],
    n_processes: int | None = None,
) -> list[R]:
    with multiprocessing.Pool(processes=n_processes) as pool:
        return pool.map(task_function, task_params)


def read_jsonl_content(filename: str | Path) -> pd.DataFrame:
    with open(filename) as file:
        content = [json.loads(l) for l in file.readlines()]  # noqa: E741
    return pd.DataFrame(content)


def read_trial_log(rundir: str | Path, log_fn: str = "trial_logs.jsonl") -> pd.DataFrame | None:
    path = Path(rundir) / log_fn
    if not path.exists():
        return None

    df = read_jsonl_content(path)
    df = normalize_drop(df, "trial_info", rename_columns=True, sep="__")
    return normalize_drop(df, "trial_value", rename_columns=True, sep="__")
    # df = df.drop(columns=["trial_info__instance", "trial_info__budget", "trial_value__time", "trial_value__status", "trial_value__starttime", "trial_value__endtime"])


def load_cfg(rundir: str | Path) -> DictConfig | None:
    config_fn = Path(rundir) / ".hydra/config.yaml"
    if not config_fn.exists():
        return None

    return OmegaConf.load(config_fn)  # type: ignore


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


def add_scenario_type(logs: pd.DataFrame, task_prefix: str = "task.") -> pd.DataFrame:
    def determine_scenario_type(x: pd.Series) -> str:
        if x[task_prefix + "is_multifidelity"] is False and x[task_prefix + "is_multiobjective"] is False:
            scenario = "blackbox"
        elif x[task_prefix + "is_multifidelity"] is True and x[task_prefix + "is_multiobjective"] is False:
            scenario = "multi-fidelity"
        elif x[task_prefix + "is_multifidelity"] is False and x[task_prefix + "is_multiobjective"] is True:
            scenario = "multi-objective"
        elif x[task_prefix + "is_multifidelity"] is True and x[task_prefix + "is_multiobjective"] is True:
            scenario = "multi-fidelity-objective"
        elif np.isnan(x[task_prefix + "is_multifidelity"]) or np.isnan(x[task_prefix + "is_multiobjective"]):
            scenario = "blackbox"
        else:
            print(
                x["problem_id"],
                x["optimizer_id"],
                x["seed"],
                x[task_prefix + "is_multifidelity"],
                type(x[task_prefix + "is_multifidelity"]),
            )
            raise ValueError("Unknown scenario")
        return scenario

    logs["scenario"] = logs.apply(determine_scenario_type, axis=1)
    return logs


def maybe_postadd_task(logs: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    index_fn = Path(__file__).parent.parent / "configs/problem/index.csv"
    if not index_fn.is_file():
        raise ValueError("Problem ids have not been indexed. Run `python -m carps.utils.index_configs`.")
    problem_index = pd.read_csv(index_fn)

    def load_task_cfg(problem_id: str) -> DictConfig:
        subset = problem_index["config_fn"][problem_index["problem_id"] == problem_id]
        if len(subset) == 0:
            raise ValueError(
                f"Can't find config_fn for {problem_id}. Maybe the index is old. Run `python -m carps.utils.index_configs` to refresh."
            )
        config_fn = subset.iloc[0]
        if not Path(config_fn).is_file():
            raise ValueError(
                f"Can't find config_fn for {problem_id}. Maybe the index is old. Run `python -m carps.utils.index_configs` to refresh."
            )
        cfg = OmegaConf.load(config_fn)
        return cfg.task

    new_logs = []
    for gid, gdf in logs.groupby(by="problem_id"):
        task_cfg = load_task_cfg(problem_id=gid)
        task_columns = [c for c in gdf.columns if c.startswith("task.")]
        if overwrite:
            task_dict = asdict(Task(**task_cfg))
            task_columns = ["task." + k for k in task_dict]

        for c in task_columns:
            key = c.split(".")[1]
            # print(task_cfg, c)
            # print(gdf[c].explode().unique())
            if overwrite or np.nan in gdf[c].explode().unique():
                v = task_cfg.get(key)
                if isinstance(v, list | ListConfig):
                    v = [v] * len(gdf)
                gdf[c] = v
        new_logs.append(gdf)
    return pd.concat(new_logs)


def filter_task_info(logs: pd.DataFrame, keep_task_columns: list[str] | None = None) -> pd.DataFrame:
    if keep_task_columns is None:
        keep_task_columns = ["n_trials"]
    keep_task_columns = [f"task.{c}" for c in keep_task_columns]
    task_cols_to_remove = [c for c in logs.columns if c.startswith("task") and c not in keep_task_columns]
    return logs.drop(columns=task_cols_to_remove)


def maybe_convert_cost_dtype(x: Any) -> tuple[float, list[float]]:
    if isinstance(x, int | float):
        return float(x)
    elif isinstance(x, str):
        return eval(x)
    else:
        assert isinstance(x, list)
        return x


def maybe_convert_cost_to_so(x: Any) -> float:
    if isinstance(x, list):
        return np.sum(x)  # TODO replace by hypervolume or similar
    else:
        return x


def convert_mixed_types_to_str(logs: pd.DataFrame, logger=None) -> pd.DataFrame:
    mixed_type_columns = logs.select_dtypes(include=["O"]).columns
    if logger:
        logger.debug(f"Goodybe all mixed data, ruthlessly converting {mixed_type_columns} to str...")
    for c in mixed_type_columns:
        # D = logs[c]
        # logs.drop(columns=c)
        if c == "cfg_str":
            continue
        logs[c] = logs[c].map(lambda x: str(x))
        logs[c] = logs[c].astype("str")
    return logs


def load_set(paths: list[str], set_id: str = "unknown") -> tuple[pd.DataFrame, pd.DataFrame]:
    logs = []
    for p in paths:
        fn = Path(p) / "trajectory.parquet"
        if not fn.is_file():
            fn = Path(p) / "logs.parquet"
        logs.append(pd.read_parquet(fn))

    df = pd.concat(logs).reset_index(drop=True)
    df_cfg = pd.concat([pd.read_parquet(Path(p) / "logs_cfg.parquet") for p in paths]).reset_index(drop=True)
    df["set"] = set_id
    return df, df_cfg


def process_logs(logs: pd.DataFrame, keep_task_columns: list[str] | None = None) -> pd.DataFrame:
    if keep_task_columns is None:
        keep_task_columns = ["task.n_trials"]
    logger.debug("Processing raw logs. Normalize n_trials and costs. Calculate trajectory (incumbent cost).")
    # logs= logs.drop(columns=["config"])
    # Filter MO costs
    logger.debug("Remove DUMMY logs...")
    logs = logs[~logs["problem_id"].str.startswith("DUMMY")]
    logs = logs[~logs["benchmark_id"].str.startswith("DUMMY")]
    logs = logs[~logs["optimizer_id"].str.startswith("DUMMY")]

    logger.debug("Handle MO costs...")
    logs["trial_value__cost_raw"] = logs["trial_value__cost"].apply(maybe_convert_cost_dtype)
    logs["trial_value__cost"] = logs["trial_value__cost_raw"].apply(maybe_convert_cost_to_so)
    logger.debug("Determine incumbent cost...")
    logs["trial_value__cost_inc"] = logs.groupby(by=["problem_id", "optimizer_id", "seed"])[
        "trial_value__cost"
    ].transform("cummin")

    logger.debug("Maybe add task info...")
    logs = maybe_postadd_task(logs)
    if "task.n_objectives" in logs:
        logs["task.is_multiobjective"] = logs["task.n_objectives"] > 1
    logger.debug("Infer scenario...")
    logs = add_scenario_type(logs)

    # Check for scalarized MO, we want to keep the cost vector
    if "trial_value__additional_info" in logs:
        ids_mo = (logs["scenario"] == "multi-objective") & (
            logs["trial_value__additional_info"].apply(lambda x: "cost" in x)
        )
        if len(ids_mo) > 0:
            logs[ids_mo]["trial_value__cost_raw"] = logs[ids_mo]["trial_value__additional_info"].apply(
                lambda x: x["cost"]
            )

    logger.debug(f"Remove task info, only keep {keep_task_columns}...")
    logs = filter_task_info(logs, keep_task_columns)

    # Convert config to object
    logger.debug("Save config as a string to avoid mixed type columns...")
    logs["trial_info__config"] = logs["trial_info__config"].apply(lambda x: str(x))

    # Add time
    logger.debug("Calculate the elapsed time...")
    logs = logs.groupby(by=["problem_id", "optimizer_id", "seed"]).apply(calc_time).reset_index(drop=True)

    logs = convert_mixed_types_to_str(logs, logger)
    logger.debug("Done 😪🙂")
    return logs


def normalize_logs(logs: pd.DataFrame) -> pd.DataFrame:
    logger.info("Start normalization...")
    logger.info("Normalize n_trials...")
    logs["n_trials_norm"] = logs.groupby("problem_id")["n_trials"].transform(normalize)
    logger.info("Normalize cost...")
    # Handle MO
    ids_mo = logs["scenario"] == "multi-objective"
    if len(ids_mo) > 0 and "hypervolume" in logs:
        hv = logs.loc[ids_mo, "hypervolume"]
        logs.loc[ids_mo, "trial_value__cost"] = -hv  # higher is better
        logs["trial_value__cost"] = logs["trial_value__cost"].astype("float64")
        logs["trial_value__cost_inc"] = logs["trial_value__cost"].transform("cummin")
    logs["trial_value__cost_norm"] = logs.groupby("problem_id")["trial_value__cost"].transform(normalize)
    logger.info("Calc normalized incumbent cost...")
    logs["trial_value__cost_inc_norm"] = logs.groupby(by=["problem_id", "optimizer_id", "seed"])[
        "trial_value__cost_norm"
    ].transform("cummin")
    if "time" not in logs:
        logs["time"] = 0
    logger.info("Normalize time...")
    logs["time_norm"] = logs.groupby("problem_id")["time"].transform(normalize)
    logs = convert_mixed_types_to_str(logs, logger)
    logger.info("Done.")
    return logs


def calc_time(D: pd.DataFrame) -> pd.Series:
    trialtime = D["trial_value__virtual_time"]
    nulltime = D["trial_value__starttime"] - D["trial_value__starttime"].min()
    trialtime_cum = trialtime.cumsum()
    elapsed = nulltime + trialtime_cum
    elapsed.name = "time"
    D["time"] = elapsed
    return D


def normalize(S: pd.Series, epsilon: float = 1e-8) -> pd.Series:
    return (S - S.min()) / (S.max() - S.min() + epsilon)


def get_interpolated_performance_df(
    logs: pd.DataFrame,
    n_points: int = 20,
    x_column: str = "n_trials_norm",
    interpolation_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Get performance dataframe for plotting.

    Interpolated at regular intervals.

    Parameters
    ----------
    logs : pd.DataFrame
        Preprocessed logs.
    n_points : int, optional
        Number of interpolation steps, by default 20
    x_column : str, optional
        The x-axis column to interpolate by, by default 'n_trials_norm'

    Raises:
    ------
    ValueError
        When x_column missing in dataframe.

    Returns:
    -------
    pd.DataFrame
        Performance data frame for plotting
    """
    if interpolation_columns is None:
        interpolation_columns = [
            "trial_value__cost",
            "trial_value__cost_norm",
            "trial_value__cost_inc",
            "trial_value__cost_inc_norm",
        ]
    logger.info("Create dataframe for neat plotting by aligning x-axis / interpolating budget.")

    if x_column not in logs:
        msg = f"x_column `{x_column}` not in logs! Did you call `carps.analysis.process_data.process_logs` on the raw logs?"
        raise ValueError(msg)

    # interpolation_columns = [
    #     c for c in logs.columns if c != x_column and c not in identifier_columns and not c.startswith("problem")]
    group_keys = ["scenario", "set", "benchmark_id", "optimizer_id", "problem_id", "seed"]
    x = np.linspace(0, 1, n_points + 1)
    D = []
    for gid, gdf in logs.groupby(by=group_keys):
        metadata = dict(zip(group_keys, gid, strict=False))
        performance_data = {}
        performance_data[x_column] = x
        for icol in interpolation_columns:
            if icol in gdf:
                xp = gdf[x_column].to_numpy()
                fp = gdf[icol].to_numpy()
                y = np.interp(x=x, xp=xp, fp=fp)
                performance_data[icol] = y
        performance_data.update(metadata)
        D.append(pd.DataFrame(performance_data))
    return pd.concat(D).reset_index(drop=True)


def load_logs(rundir: str):
    logs_fn = Path(rundir) / "logs.csv"
    logs_cfg_fn = logs_fn.parent / "logs_cfg.csv"

    logger.info(f"Load logs from `{logs_fn}` and associated configs from {logs_cfg_fn}. Preprocess logs.")

    if not logs_fn.is_file() or not logs_cfg_fn.is_file():
        msg = f"No logs found at rundir '{rundir}'. If you used the file logger, you can gather the data with `python -m carps.analysis.gather_data <rundir>`."
        raise RuntimeError(msg)

    df = pd.read_csv(logs_fn)
    df = normalize_logs(df)
    df_cfg = pd.read_csv(logs_cfg_fn)
    return df, df_cfg


# NOTE(eddiebergman): Use `n_processes=None` as default, which uses `os.cpu_count()` in `Pool`
def filelogs_to_df(
    rundir: str, log_fn: str = "trial_logs.jsonl", n_processes: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Get rundirs from {rundir}...")
    rundirs = get_run_dirs(rundir)
    logger.info(f"Found {len(rundirs)} runs. Load data...")
    partial_load_log = partial(load_log, log_fn=log_fn)
    results = map_multiprocessing(partial_load_log, rundirs, n_processes=n_processes)
    df = pd.concat(results).reset_index(drop=True)
    logger.info("Done. Do some preprocessing...")
    df_cfg = pd.DataFrame([{"cfg_fn": k, "cfg_str": v} for k, v in df["cfg_str"].unique()])
    df_cfg.loc[:, "experiment_id"] = np.arange(0, len(df_cfg))
    df["experiment_id"] = df["cfg_fn"].apply(lambda x: np.where(df_cfg["cfg_fn"].to_numpy() == x)[0][0])
    df_cfg.loc[:, "cfg_str"] = df_cfg["cfg_str"].apply(lambda x: x.replace("\n", "\\n"))
    del df["cfg_str"]
    del df["cfg_fn"]
    logger.info("Done. Saving to file...")
    # df = df.map(lambda x: x if not isinstance(x, list) else str(x))
    df.to_csv(Path(rundir) / "logs.csv", index=False)
    df_cfg.to_csv(Path(rundir) / "logs_cfg.csv", index=False)
    df = convert_mixed_types_to_str(df)
    df_cfg = convert_mixed_types_to_str(df_cfg)
    df.to_parquet(Path(rundir) / "logs.parquet", index=False)
    df_cfg.to_parquet(Path(rundir) / "logs_cfg.parquet", index=False)
    logger.info("Done. 😊")
    return df, df_cfg


if __name__ == "__main__":
    fire.Fire(filelogs_to_df)
