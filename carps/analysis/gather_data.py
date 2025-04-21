"""Gather data from file logs and preprocess."""

from __future__ import annotations

import ast
import json
import logging
import multiprocessing
from collections.abc import Callable, Iterable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

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
    from carps.objective_functions.objective_function import ObjectiveFunction

setup_logging()
logger = get_logger(__file__)


setup_globals()


def glob_trial_logs(p: str | Path) -> list[Path]:
    """Glob trial logs.

    Args:
        p (str | Path): Rundir to perform the globbing in.

    Returns:
        list[Path]: List of paths to trial logs. The paths are relative to the rundir. The trial log path's location is
            the rundir of a single run.
    """
    return list(Path(p).glob("**/trial_logs.jsonl"))


def get_run_dirs(outdir: str) -> list[Path]:
    """Get run directories.

    Args:
        outdir (str): Output directory.

    Returns:
        list[Path]: List of paths to run directories.
    """
    opt_paths = list(Path(outdir).glob("*/*"))
    with multiprocessing.Pool() as pool:
        triallog_files = pool.map(glob_trial_logs, opt_paths)
    if len(triallog_files) == 0:
        raise ValueError("No trial logs found.")
    return [f.parent for f in np.concatenate(triallog_files)]  # type: ignore[attr-defined]


def annotate_with_cfg(
    df: pd.DataFrame,
    cfg: DictConfig,
    config_keys: list[str],
    config_keys_forbidden: list[str] | None = None,
) -> pd.DataFrame:
    """Annotate data frame with config.

    Args:
        df (pd.DataFrame): Data frame.
        cfg (DictConfig): Config.
        config_keys (list[str]): Config keys to annotate.
        config_keys_forbidden (list[str] | None, optional): Forbidden config keys. Defaults to None.

    Returns:
        pd.DataFrame: Annotated data frame.
    """
    if config_keys_forbidden is None:
        config_keys_forbidden = []
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    flat_cfg = pd.json_normalize(cfg_resolved, sep=".").iloc[0].to_dict()  # type: ignore
    for k, v in flat_cfg.items():
        if np.any([k.startswith(c) for c in config_keys]) and not np.any([c in k for c in config_keys_forbidden]):
            value = v
            if isinstance(v, list | ListConfig):
                value = [v] * len(df)
            df[k] = value
    return df


def get_Y(X: np.ndarray, objective_function: ObjectiveFunction) -> np.ndarray:  # noqa: N802, N803
    """Get objective function values.

    Beware, remember runtime when objective_function is not synthetic, a table or a surrogate.

    Args:
        X (np.ndarray): Design points.
        objective_function (ObjectiveFunction): ObjectiveFunction instance.

    Returns:
        np.ndarray: Objective function values.
    """
    return np.array(
        [
            (
                objective_function.evaluate(
                    trial_info=TrialInfo(
                        config=Configuration(configuration_space=objective_function.configspace, vector=x)
                    )
                ).cost
            )
            for x in X
        ]
    )


def join_df(df1: pd.DataFrame, df2: pd.DataFrame, on: str = "n_trials") -> pd.DataFrame:
    """Join two data frames.

    Args:
        df1 (pd.DataFrame): Data frame 1.
        df2 (pd.DataFrame): Data frame 2.
        on (str, optional): Column to join on. Defaults to "n_trials".

    Returns:
        pd.DataFrame: Joined data frame.
    """
    df1.set_index(on)
    return df1.join(df2.set_index(on), on=on)


def load_log(rundir: str | Path, log_fn: str = "trial_logs.jsonl") -> pd.DataFrame:
    """Load log / one optimization run from rundir and annotate with config.

    Args:
        rundir (str | Path): Run directory.
        log_fn (str, optional): Log filename. Defaults to "trial_logs.jsonl".

    Returns:
        pd.DataFrame: Data frame for one optimization run.
    """
    df = read_trial_log(rundir, log_fn=log_fn)  # noqa: PD901
    if df is None:
        # raise NotImplementedError("No idea what should happen here!?")
        return pd.DataFrame()

    cfg = load_cfg(rundir)
    if cfg is not None:
        config_fn = str(Path(rundir) / ".hydra/config.yaml")
        cfg_str = OmegaConf.to_yaml(cfg=cfg)
        df["cfg_fn"] = config_fn
        df["cfg_str"] = [(config_fn, cfg_str)] * len(df)

        config_keys = [
            "benchmark_id",
            "task_id",
            "task_type",
            "subset_id",
            "benchmark",
            "task",
            "seed",
            "optimizer_id",
        ]
        config_keys_forbidden = ["_target_", "_partial_"]
        df = annotate_with_cfg(df=df, cfg=cfg, config_keys=config_keys, config_keys_forbidden=config_keys_forbidden)  # noqa: PD901
    else:
        config_fn = "no_hydra_config"
        cfg_str = ""
        df["cfg_fn"] = config_fn
        df["cfg_str"] = [(config_fn, cfg_str)] * len(df)

    if "problem.function.seed" in df:
        df = df.drop(columns=["problem.function.seed"])  # noqa: PD901

    if "problem.function.dim" in df:
        df = df.rename(columns={"task.function.dim": "dim"})  # noqa: PD901

    return process_logs(df)


T = TypeVar("T")
R = TypeVar("R")


def map_multiprocessing(
    task_function: Callable[[T], R],
    task_params: Iterable[T],
    n_processes: int | None = None,
) -> list[R]:
    """Map function to iterable with multiprocessing.

    Args:
        task_function (Callable[[T], R]): Task function.
        task_params (Iterable[T]): Task parameters.
        n_processes (int | None, optional): Number of processes. Defaults to None.

    Returns:
        list[R]: Results.
    """
    with multiprocessing.Pool(processes=n_processes) as pool:
        return pool.map(task_function, task_params)


def read_jsonl_content(filename: str | Path) -> pd.DataFrame:
    """Read JSONL content.

    In JSONL, each line is a json object.

    Args:
        filename (str | Path): Filename.

    Returns:
        pd.DataFrame: Data frame.
    """
    with open(filename) as file:
        content = [json.loads(l) for l in file.readlines()]  # noqa: E741
    return pd.DataFrame(content)


def read_trial_log(rundir: str | Path, log_fn: str = "trial_logs.jsonl") -> pd.DataFrame | None:
    """Read trial log.

    Args:
        rundir (str | Path): Run directory.
        log_fn (str, optional): Log filename. Defaults to "trial_logs.jsonl".

    Returns:
        pd.DataFrame | None: Data frame for one optimization run.
    """
    path = Path(rundir) / log_fn
    if not path.exists():
        return None

    df = read_jsonl_content(path)  # noqa: PD901
    df = normalize_drop(df, "trial_info", rename_columns=True, sep="__")  # noqa: PD901
    return normalize_drop(df, "trial_value", rename_columns=True, sep="__")


def load_cfg(rundir: str | Path) -> DictConfig | None:
    """Load hydra config holding the experiment settings.

    Args:
        rundir (str | Path): Run directory.

    Returns:
        DictConfig | None: Config when config file exists, None otherwise.
    """
    config_fn = Path(rundir) / ".hydra/config.yaml"
    if not config_fn.exists():
        return None

    return OmegaConf.load(config_fn)  # type: ignore


def normalize_drop(df: pd.DataFrame, key: str, rename_columns: bool = False, sep: str = ".") -> pd.DataFrame:  # noqa: FBT001, FBT002
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
    """Add n_trials column to data frame if missing.

    Args:
        df (pd.DataFrame): Data frame.
        n_initial_design (int): Initial design size.
        counter_key (str, optional): Counter key. Defaults to "n_calls".

    Returns:
        pd.DataFrame: Data frame with n_trials column.
    """
    if "n_trials" not in df:
        df["n_trials"] = df[counter_key] + n_initial_design  # n_trials is 1-based
    return df


def add_task_type(logs: pd.DataFrame, task_prefix: str = "task.") -> pd.DataFrame:
    """Add task type to logs.

    Args:
        logs (pd.DataFrame): Logs.
        task_prefix (str, optional): Task prefix. Defaults to "task.". Set it such that it matches the node task in the
            hydra config.

    Returns:
        pd.DataFrame: Logs with task type.
    """

    def determine_task_type(x: pd.Series) -> str:
        get_mf = x.get(task_prefix + "is_multifidelity", False)
        get_mo = x.get(task_prefix + "is_multiobjective", False)
        if get_mf is False and get_mo is False:
            task_type = "blackbox"
        elif get_mf is True and get_mo is False:
            task_type = "multi-fidelity"
        elif get_mf is False and get_mo is True:
            task_type = "multi-objective"
        elif get_mf is True and get_mo is True:
            task_type = "multi-fidelity-objective"
        elif np.isnan(get_mf) or np.isnan(get_mo):
            task_type = "blackbox"
        else:
            print(
                x["task_id"],
                x["optimizer_id"],
                x["seed"],
                get_mf,
                type(get_mf),
            )
            raise ValueError("Unknown task_type")
        return task_type

    logs["task_type"] = logs.apply(determine_task_type, axis=1)
    return logs


def load_task_cfg(task_id: str, task_index: pd.DataFrame) -> DictConfig:
    """Load task config from index by id.

    Args:
        task_id (str): Task id.
        task_index (pd.DataFrame): Task index. Generated by `python -m carps.utils.index_configs`.

    Returns:
        DictConfig: Task config.
    """
    subset = task_index["config_fn"][task_index["task_id"] == task_id]
    if len(subset) == 0:
        raise ValueError(
            f"Can't find config_fn for {task_id}. Maybe the index is old. Run "
            "`python -m carps.utils.index_configs` to refresh."
        )
    config_fn = subset.iloc[0]
    if not Path(config_fn).is_file():
        raise ValueError(
            f"Can't find config_fn for {task_id}. Maybe the index is old. Run "
            "`python -m carps.utils.index_configs` to refresh."
        )
    cfg = OmegaConf.load(config_fn)
    return cfg.task


def maybe_postadd_task(logs: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:  # noqa: FBT001, FBT002
    """Maybe add task columns to logs.

    Args:
        logs (pd.DataFrame): Raw logs.
        overwrite (bool, optional): Overwrite existing, logged values from task configs. Defaults to False.

    Returns:
        pd.DataFrame: Logs with task
    """
    if "task_id" not in logs:
        logger.debug("No task_id in logs. Can't add task info.")
        return logs
    index_fn = Path(__file__).parent.parent / "configs/task/index.csv"
    if not index_fn.is_file():
        raise ValueError("ObjectiveFunction ids have not been indexed. Run `python -m carps.utils.index_configs`.")
    task_index = pd.read_csv(index_fn)

    new_logs = []
    for gid, gdf in logs.groupby(by="task_id"):
        task_cfg = load_task_cfg(task_id=gid, task_index=task_index)
        task_columns = [c for c in gdf.columns if c.startswith("task.")]
        if overwrite:
            task_dict = asdict(Task(**task_cfg))
            task_columns = ["task." + k for k in task_dict]

        for c in task_columns:
            key = c.split(".")[1]
            # print(task_cfg, c)
            # print(gdf[c].explode().unique())
            if overwrite or gdf[c].explode().isna().any():
                v = task_cfg.get(key)
                if isinstance(v, list | ListConfig):
                    v = [v] * len(gdf)
                gdf[c] = v
        new_logs.append(gdf)
    return pd.concat(new_logs)


def filter_task_info(logs: pd.DataFrame, keep_task_columns: list[str] | None = None) -> pd.DataFrame:
    """Filter task info columns if too many columns.

    Args:
        logs (pd.DataFrame): Logs.
        keep_task_columns (list[str] | None, optional): Columns to keep. Defaults to None -> keep only
        `task.optimization_resources.n_trials`.

    Returns:
        pd.DataFrame: Filtered logs.
    """
    if keep_task_columns is None:
        keep_task_columns = ["task.optimization_resources.n_trials"]
    keep_task_columns = [f"task.{c}" for c in keep_task_columns]
    task_cols_to_remove = [c for c in logs.columns if c.startswith("task.") and c not in keep_task_columns]
    return logs.drop(columns=task_cols_to_remove)


def maybe_convert_cost_dtype(x: int | float | str | list) -> float | list[float]:
    """Maybe convert cost dtype.

    Args:
        x (Any): Cost.

    Returns:
        float | list[float]: Cost(s).
    """
    if isinstance(x, int | float):
        return float(x)
    if isinstance(x, str):
        return eval(x)  # noqa: S307
    assert isinstance(x, list)
    return x


def maybe_convert_cost_to_so(x: float | list | np.ndarray) -> float:
    """Maybe convert cost to single-objective if cost is a vector by summation.

    TODO: Replace by hypervolume or similar.

    Args:
        x (float | Sequence[float]): Cost (vector).

    Raises:
        ValueError: Unknown cost type.

    Returns:
        float: Single-objective cost or aggregated cost.
    """
    if isinstance(x, list | np.ndarray):
        return np.sum(x)
    if isinstance(x, dict):
        assert len(x.values()) == 1
        # Most likely comes from database
        # {'cost': '[-8.70692741103592, 4.457404074206716]'}
        value = next(iter(x.values()))
        if isinstance(value, str):
            value = ast.literal_eval(value)
            if isinstance(value, list):
                return np.sum(value)
        if isinstance(value, float | int):
            return value
    if isinstance(x, float):
        return x
    raise ValueError(f"Unknown cost type {type(x)}. Supported are float, list, np.ndarray.")


def convert_mixed_types_to_str(logs: pd.DataFrame, logger: logging.Logger | None = None) -> pd.DataFrame:
    """Convert mixed type columns to str.

    Necessary to be able to write a parquet file.

    Args:
        logs (pd.DataFrame): Logs.
        logger (logging.Logger, optional): Logger. Defaults to None.

    Returns:
        pd.DataFrame: Logs with mixed type columns converted
    """
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
    """Load a set of logs.

    Args:
        paths (list[str]): List of paths to logs.
        set_id (str, optional): Set id. Defaults to "unknown".

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Logs and logs_cfg.
    """
    logs = []
    for p in paths:
        fn = Path(p) / "trajectory.parquet"
        if not fn.is_file():
            fn = Path(p) / "logs.parquet"
        logs.append(pd.read_parquet(fn))

    df = pd.concat(logs).reset_index(drop=True)  # noqa: PD901
    df_cfg = pd.concat([pd.read_parquet(Path(p) / "logs_cfg.parquet") for p in paths]).reset_index(drop=True)
    df["set"] = set_id
    return df, df_cfg


def process_logs(logs: pd.DataFrame, keep_task_columns: list[str] | None = None) -> pd.DataFrame:
    """Process raw logs.

    Clean, determine incumbent cost, maybe add metadata.

    Args:
        logs (pd.DataFrame): Raw logs.
        keep_task_columns (list[str] | None, optional): Columns to keep. Defaults to None.

    Returns:
        pd.DataFrame: Processed logs.
    """
    if keep_task_columns is None:
        keep_task_columns = ["task.optimization_resources.n_trials"]
    logger.debug("Processing raw logs. Normalize n_trials and costs. Calculate trajectory (incumbent cost).")
    # logs= logs.drop(columns=["config"])
    # Filter MO costs
    logger.debug("Remove DUMMY logs...")
    if "task_id" in logs:
        logs = logs[~logs["task_id"].str.startswith("DUMMY")]
    if "benchmark_id" in logs:
        logs = logs[~logs["benchmark_id"].str.startswith("DUMMY")]
    if "optimizer_id" in logs:
        logs = logs[~logs["optimizer_id"].str.startswith("DUMMY")]

    if "experiment_id" in logs:  # noqa: SIM108
        # Logs come from database
        grouper_keys = ["experiment_id"]
    else:
        # Logs come from file
        grouper_keys = ["task_id", "optimizer_id", "seed"]

    logger.debug("Handle MO costs...")
    logs["trial_value__cost_raw"] = logs["trial_value__cost"].apply(maybe_convert_cost_dtype)
    logs["trial_value__cost"] = logs["trial_value__cost_raw"].apply(maybe_convert_cost_to_so)
    logger.debug("Determine incumbent cost...")
    logs["trial_value__cost_inc"] = logs.groupby(by=grouper_keys)["trial_value__cost"].transform("cummin")

    logger.debug("Maybe add task info...")
    logs = maybe_postadd_task(logs)
    if "task.output_space.n_objectives" in logs:
        logs["task.is_multiobjective"] = logs["task.output_space.n_objectives"] > 1
    logger.debug("Infer task_type...")
    if "scenario" in logs:
        logs = logs.rename(columns={"scenario": "task_type"})
    logs = add_task_type(logs)

    # Check for scalarized MO, we want to keep the cost vector
    if "trial_value__additional_info" in logs:
        ids_mo = (logs["task_type"] == "multi-objective") & (
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
    logs = logs.groupby(by=grouper_keys).apply(calc_time, include_groups=False).reset_index(drop=False)
    logs = convert_mixed_types_to_str(logs, logger)
    logger.debug("Done 😪🙂")
    return logs


def normalize_logs(logs: pd.DataFrame) -> pd.DataFrame:
    """Normalize logs per task.

    Args:
        logs (pd.DataFrame): Raw logs.

    Returns:
        pd.DataFrame: Normalized logs
    """
    logger.info("Start normalization...")
    logger.info("Normalize n_trials...")
    logs["n_trials_norm"] = logs.groupby("task_id")["n_trials"].transform(normalize)
    logger.info("Normalize cost...")
    # Handle MO
    ids_mo = logs["task_type"] == "multi-objective"
    if len(ids_mo) > 0 and "hypervolume" in logs:
        hv = logs.loc[ids_mo, "hypervolume"]
        logs.loc[ids_mo, "trial_value__cost"] = -hv  # higher is better
        logs["trial_value__cost"] = logs["trial_value__cost"].astype("float64")
        logs["trial_value__cost_inc"] = logs["trial_value__cost"].transform("cummin")
    logs["trial_value__cost_norm"] = logs.groupby("task_id")["trial_value__cost"].transform(normalize)
    logger.info("Calc normalized incumbent cost...")

    # logs["trial_value__cost_log"] = logs["trial_value__cost"].apply(lambda x: np.log(x + 1e-10))
    logs["trial_value__cost_log"] = logs.groupby(by=["task_id"])["trial_value__cost"].transform(
        lambda x: np.log(x - x.min() + 1e-10)
    )
    logs["trial_value__cost_inc_log"] = logs.groupby(by=["task_id", "optimizer_id", "seed"])[
        "trial_value__cost_log"
    ].transform("cummin")
    logs["trial_value__cost_log_norm"] = logs.groupby("task_id")["trial_value__cost_log"].transform(normalize)
    logs["trial_value__cost_inc_log_norm"] = logs.groupby(by=["task_id", "optimizer_id", "seed"])[
        "trial_value__cost_log_norm"
    ].transform("cummin")

    logs["trial_value__cost_inc_norm"] = logs.groupby(by=["task_id", "optimizer_id", "seed"])[
        "trial_value__cost_norm"
    ].transform("cummin")
    logs["trial_value__cost_inc_norm_log"] = logs["trial_value__cost_inc_norm"].apply(lambda x: np.log(x + 1e-10))
    if "time" not in logs:
        logs["time"] = 0
    logger.info("Normalize time...")
    logs["time_norm"] = logs.groupby("task_id")["time"].transform(normalize)
    logs = convert_mixed_types_to_str(logs, logger)
    logger.info("Done.")
    return logs


def calc_time(D: pd.DataFrame) -> pd.Series:  # noqa: N803
    """Calculate time elapsed.

    Args:
        D (pd.DataFrame): Logs for a single task, optimizer, seed.

    Returns:
        pd.Series: D with time elapsed as "time" column.
    """
    trialtime = D["trial_value__virtual_time"]
    nulltime = D["trial_value__starttime"] - D["trial_value__starttime"].min()
    trialtime_cum = trialtime.cumsum()
    elapsed = nulltime + trialtime_cum
    elapsed.name = "time"
    D["time"] = elapsed
    return D


def normalize(S: pd.Series, epsilon: float = 1e-8) -> pd.Series:  # noqa: N803
    """Normalize series.

    Args:
        S (pd.Series): Series.
        epsilon (float, optional): Epsilon to avoid division by zero. Defaults to 1e-8.

    Returns:
        pd.Series: Normalized series.
    """
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
            "trial_value__cost_inc_log",
            "trial_value__cost_inc_log_norm",
            "trial_value__cost_inc_norm",
            "trial_value__cost_inc_norm_log",
        ]
    logger.info("Create dataframe for neat plotting by aligning x-axis / interpolating budget.")

    if x_column not in logs:
        msg = (
            f"x_column `{x_column}` not in logs! "
            "Did you call `carps.analysis.process_data.process_logs` on the raw logs?"
        )
        raise ValueError(msg)

    # interpolation_columns = [
    #     c for c in logs.columns if c != x_column and c not in identifier_columns and not c.startswith("task")]
    group_keys = ["task_type", "set", "benchmark_id", "optimizer_id", "task_id", "seed"]
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


def load_logs(rundir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load logs from rundir and associated configs.

    Args:
        rundir (str): Run directory containing subdirectories with single runs. Can have any structure and levels.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Logs and logs_cfg.
    """
    logs_fn = Path(rundir) / "logs.csv"
    logs_cfg_fn = logs_fn.parent / "logs_cfg.csv"

    logger.info(f"Load logs from `{logs_fn}` and associated configs from {logs_cfg_fn}. Preprocess logs.")

    if not logs_fn.is_file() or not logs_cfg_fn.is_file():
        msg = (
            f"No logs found at rundir '{rundir}'. "
            "If you used the file logger, you can gather the data with `python -m carps.analysis.gather_data <rundir>`."
        )
        raise RuntimeError(msg)

    df = pd.read_csv(logs_fn)  # noqa: PD901
    df = normalize_logs(df)  # noqa: PD901
    df_cfg = pd.read_csv(logs_cfg_fn)
    return df, df_cfg


def rename_legacy(logs: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy columns.

    Args:
        logs (pd.DataFrame): Logs.

    Returns:
        pd.DataFrame: Logs with renamed columns.
    """
    columns = {
        "problem_id": "task_id",
        "scenario": "task_type",
    }
    return logs.rename(columns=columns)


# NOTE(eddiebergman): Use `n_processes=None` as default, which uses `os.cpu_count()` in `Pool`
def filelogs_to_df(
    rundir: str | list[str], log_fn: str = "trial_logs.jsonl", n_processes: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load logs from file and preprocess.

    Will collect all results from all runs contained in `rundir`.

    Parameters
    ----------
    rundir : str | Path | list[str]
        Directory containing logs.
    log_fn : str, optional
        Filename of the log file, by default "trial_logs.jsonl"
    n_processes : int | None, optional
        Number of processes to use for multiprocessing, by default None

    Returns.
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Logs and config data frames.
    """
    if isinstance(rundir, str):
        rundir = [rundir]
    rundirs_list = rundir
    df_list = []
    for rundir in rundirs_list:
        logger.info(f"Get rundirs from {rundir}...")
        rundirs = get_run_dirs(rundir)
        logger.info(f"Found {len(rundirs)} runs. Load data...")
        partial_load_log = partial(load_log, log_fn=log_fn)
        results = map_multiprocessing(partial_load_log, rundirs, n_processes=n_processes)
        df = pd.concat(results).reset_index(drop=True)  # noqa: PD901
        logger.info("Done. Do some preprocessing...")
        df_cfg = pd.DataFrame([{"cfg_fn": k, "cfg_str": v} for k, v in df["cfg_str"].unique()])
        df_cfg.loc[:, "experiment_id"] = np.arange(0, len(df_cfg))
        df["experiment_id"] = df["cfg_fn"].apply(
            lambda x, df_cfg=df_cfg: np.where(df_cfg["cfg_fn"].to_numpy() == x)[0][0]
        )
        df_cfg.loc[:, "cfg_str"] = df_cfg["cfg_str"].apply(lambda x: x.replace("\n", "\\n"))
        del df["cfg_str"]
        del df["cfg_fn"]
        df_list.append(df)
    df = pd.concat(df_list).reset_index(drop=True)  # noqa: PD901
    logger.info("Done. Saving to file...")
    # df = df.map(lambda x: x if not isinstance(x, list) else str(x))
    df.to_csv(Path(rundir) / "logs.csv", index=False)
    df_cfg.to_csv(Path(rundir) / "logs_cfg.csv", index=False)
    df = convert_mixed_types_to_str(df)  # noqa: PD901
    df_cfg = convert_mixed_types_to_str(df_cfg)
    df.to_parquet(Path(rundir) / "logs.parquet", index=False)
    df_cfg.to_parquet(Path(rundir) / "logs_cfg.parquet", index=False)
    logger.info("Done. 😊")
    return df, df_cfg


if __name__ == "__main__":
    fire.Fire(filelogs_to_df)
