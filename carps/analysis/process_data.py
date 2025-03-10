"""Process raw logs and configs for analysis."""

from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

from carps.utils.loggingutils import get_logger, setup_logging

setup_logging()
logger = get_logger(__file__)


def add_scenario_type(logs: pd.DataFrame) -> pd.DataFrame:
    """Add scenario type to logs.

    Parameters
    ----------
    logs : pd.DataFrame
        Raw logs.

    Returns:
    -------
    pd.DataFrame
        Logs with scenario type.
    """

    def determine_scenario_type(x: pd.Series) -> str:
        if x["task.input_space.fidelity_space.is_multifidelity"] is False and x["task.is_multiobjective"] is False:
            scenario = "blackbox"
        elif x["task.input_space.fidelity_space.is_multifidelity"] is True and x["task.is_multiobjective"] is False:
            scenario = "multi-fidelity"
        elif x["task.input_space.fidelity_space.is_multifidelity"] is False and x["task.is_multiobjective"] is True:
            scenario = "multi-objective"
        elif x["task.input_space.fidelity_space.is_multifidelity"] is True and x["task.is_multiobjective"] is True:
            scenario = "multi-fidelity-objective"
        elif np.isnan(x["task.input_space.fidelity_space.is_multifidelity"]) or np.isnan(x["task.is_multiobjective"]):
            scenario = "blackbox"
        else:
            print(
                x["task_id"],
                x["optimizer_id"],
                x["seed"],
                x["task.input_space.fidelity_space.is_multifidelity"],
                type(x["task.input_space.fidelity_space.is_multifidelity"]),
            )
            raise ValueError("Unknown scenario")
        return scenario

    logs["scenario"] = logs.apply(determine_scenario_type, axis=1)
    return logs


def maybe_postadd_task(logs: pd.DataFrame) -> pd.DataFrame:
    """Post-add task columns to logs.

    Parameters
    ----------
    logs : pd.DataFrame
        Raw logs.

    Returns:
    -------
    pd.DataFrame
        Logs with task columns.
    """
    index_fn = Path(__file__).parent.parent / "configs/task/index.csv"
    if not index_fn.is_file():
        raise ValueError("Task ids have not been indexed. Run `python -m carps.utils.index_configs`.")
    task_index = pd.read_csv(index_fn)

    def load_task_cfg(task_id: str) -> DictConfig:
        config_fn = task_index["config_fn"][task_index["task_id"] == task_id].iloc[0]
        if not Path(config_fn).is_file():
            raise ValueError("Maybe the index is old. Run `python -m carps.utils.index_configs` to refresh.")
        cfg = OmegaConf.load(config_fn)
        return cfg.task

    new_logs = []
    for gid, gdf in logs.groupby(by="task_id"):
        task_cfg = load_task_cfg(task_id=gid)
        task_columns = [c for c in gdf.columns if c.startswith("task.")]
        for c in task_columns:
            key = c.split(".")[1]
            if np.nan in gdf[c].unique():
                print(c, key, task_cfg, gid)
                v = task_cfg.get(key)
                if isinstance(v, list | ListConfig):
                    v = [v] * len(gdf)
                gdf[c] = v
        new_logs.append(gdf)
    return pd.concat(new_logs)


def process_logs(logs: pd.DataFrame) -> pd.DataFrame:
    """Process raw logs.

    Normalize n_trials and costs. Calculate trajectory (incumbent cost). Maybe add scenario.

    Parameters
    ----------
    logs : pd.DataFrame
        Raw logs.

    Returns:
    -------
    pd.DataFrame
        Processed logs
    """
    logger.info("Processing raw logs. Normalize n_trials and costs. Calculate trajectory (incumbent cost).")
    # logs= logs.drop(columns=["config"])
    # Filter MO costs
    logs = logs[~logs["task_id"].str.startswith("DUMMY")]
    logs = logs[~logs["benchmark_id"].str.startswith("DUMMY")]
    logs = logs[~logs["optimizer_id"].str.startswith("DUMMY")]
    logs["trial_value__cost"] = logs["trial_value__cost"].apply(lambda x: x if isinstance(x, float) else eval(x))  # noqa: S307
    logs = logs[logs["trial_value__cost"].apply(lambda x: isinstance(x, float))]
    logs["trial_value__cost"] = logs["trial_value__cost"].apply(float)
    logs["n_trials_norm"] = logs.groupby("task_id")["n_trials"].transform(normalize)
    logs["trial_value__cost_norm"] = logs.groupby("task_id")["trial_value__cost"].transform(normalize)
    logs["trial_value__cost_inc"] = logs.groupby(by=["task_id", "optimizer_id", "seed"])["trial_value__cost"].transform(
        "cummin"
    )
    logs["trial_value__cost_inc_norm"] = logs.groupby(by=["task_id", "optimizer_id", "seed"])[
        "trial_value__cost_norm"
    ].transform("cummin")
    logs = maybe_postadd_task(logs)
    if "task.output_space.n_objectives" in logs:
        logs["task.is_multiobjective"] = logs["task.output_space.n_objectives"] > 1
    logs = add_scenario_type(logs)

    # Add time
    logs = logs.groupby(by=["task_id", "optimizer_id", "seed"]).apply(calc_time).reset_index(drop=True)
    logs["time_norm"] = logs.groupby("task_id")["time"].transform(normalize)
    return logs


def calc_time(D: pd.DataFrame) -> pd.Series:  # noqa: N803
    """Calculate time elapsed.

    Parameters
    ----------
    D : pd.DataFrame
        Logs for a single task, optimizer, seed.

    Returns:
    -------
    pd.Series
        D with time elapsed as "time" column.
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

    Parameters
    ----------
    S : pd.Series
        Series to normalize.
    epsilon : float, optional
        Small value to avoid division by zero, by default 1e-8.

    Returns:
    -------
    pd.Series
        Normalized series
    """
    return (S - S.min()) / (S.max() - S.min() + epsilon)


def get_interpolated_performance_df(
    logs: pd.DataFrame, n_points: int = 20, x_column: str = "n_trials_norm"
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
    logger.info("Create dataframe for neat plotting by aligning x-axis / interpolating budget.")

    if x_column not in logs:
        msg = (
            f"x_column `{x_column}` not in logs! Did you call "
            "`carps.analysis.process_data.process_logs` on the raw logs?"
        )
        raise ValueError(msg)

    interpolation_columns = [
        "trial_value__cost",
        "trial_value__cost_norm",
        "trial_value__cost_inc",
        "trial_value__cost_inc_norm",
    ]
    # interpolation_columns = [
    #     c for c in logs.columns if c != x_column and c not in identifier_columns and not c.startswith("task")]
    group_keys = ["scenario", "benchmark_id", "optimizer_id", "task_id", "seed"]
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
    """Load logs and associated configs from rundir.

    Parameters
    ----------
    rundir : str
        Run directory.

    Returns:
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Logs, configs
    """
    logs_fn = Path(rundir) / "logs.csv"
    logs_cfg_fn = logs_fn.parent / "logs_cfg.csv"

    logger.info(f"Load logs from `{logs_fn}` and associated configs from {logs_cfg_fn}. Preprocess logs.")

    if not logs_fn.is_file() or not logs_cfg_fn.is_file():
        msg = (
            f"No logs found at rundir '{rundir}'. "
            "If you used the file logger, you can gather the data "
            "with `python -m carps.analysis.gather_data <rundir>`."
        )
        raise RuntimeError(msg)

    df = pd.read_csv(logs_fn)  # noqa: PD901
    df = process_logs(df)  # noqa: PD901
    df_cfg = pd.read_csv(logs_cfg_fn)
    return df, df_cfg


if __name__ == "__main__":
    fire.Fire(load_logs)
