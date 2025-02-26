"""Check missing runs and regenerate runcommands for missing or truncated runs."""

from __future__ import annotations

from enum import Enum, auto
from multiprocessing import Pool
from pathlib import Path

import fire
import pandas as pd
from omegaconf import OmegaConf

from carps.analysis.gather_data import read_jsonl_content
from carps.utils.loggingutils import get_logger, setup_logging

setup_logging()
logger = get_logger(__file__)


class RunStatus(Enum):
    """Enum for the status of a run."""

    COMPLETED = auto()
    MISSING = auto()
    TRUNCATED = auto()


def get_experiment_status(path: Path) -> dict:
    """Get the status of an experiment from its hydra config file.

    Args:
        path (Path): Path to the hydra config file.
    """
    status = RunStatus.MISSING

    cfg = OmegaConf.load(path)
    n_trials = cfg.task.n_trials
    trial_logs_fn = path.parent.parent / "trial_logs.jsonl"
    if trial_logs_fn.is_file():
        trial_logs = read_jsonl_content(str(trial_logs_fn))
        n_trials_done = trial_logs["n_trials"].max()
        status = RunStatus.COMPLETED if n_trials >= n_trials_done else RunStatus.TRUNCATED

    overrides = OmegaConf.load(path.parent / "overrides.yaml")
    # TODO maybe filter cluster
    return {
        "status": status.name,
        "benchmark_id": cfg.benchmark_id,
        "problem_id": cfg.problem_id,
        "optimizer_id": cfg.optimizer_id,
        "seed": cfg.seed,
        "overrides": " ".join(overrides),
    }


def check_missing(rundir: str | Path, n_processes: int = 4) -> pd.DataFrame:
    """Check missing runs in the given rundir.

    Args:
        rundir (str | Path): Path to the rundir.
        n_processes (int, optional): Number of processes to use. Defaults to 4.

    Returns:
        pd.DataFrame: DataFrame containing the status of the runs. This is also saved to 'runstatus.csv'.
    """
    rundir = Path(rundir)
    paths = rundir.glob("**/.hydra/config.yaml")
    with Pool(processes=n_processes) as pool:
        data = pool.map(get_experiment_status, paths)
    data_df = pd.DataFrame(data)
    data_df.to_csv("runstatus.csv", index=False)
    return data


def generate_commands(missing_data: pd.DataFrame, runstatus: RunStatus, rundir: str = ".") -> None:
    """Generate runcommands for missing or truncated runs.

    Args:
        missing_data (pd.DataFrame): DataFrame containing the status of the runs.
        runstatus (RunStatus): Status of the runs to generate commands for.
        rundir (str, optional): Path to the rundir. Defaults to ".".
    """
    logger.info(f"Regenerate commands for {runstatus.name} runs...")
    data = missing_data
    missing = data[data["status"].isin([runstatus.name])]
    runcommands = []
    for _gid, gdf in missing.groupby(by=["optimizer_id", "problem_id"]):
        seeds = list(gdf["seed"].unique())
        seeds.sort()
        overrides = gdf["overrides"].iloc[0].split(" ")
        overrides = [o for o in overrides if "seed" not in o]
        overrides.append(f"seed={','.join(str(s) for s in seeds)} -m")
        override = " ".join(overrides)
        runcommand = f"python -m carps.run {override}\n"
        runcommands.append(runcommand)
    runcommand_fn = Path(rundir) / f"runcommands_{runstatus.name}.sh"
    with open(runcommand_fn, "w") as file:
        file.writelines(runcommands)
    logger.info(f"Done! Regenerated runcommands at {runcommand_fn}.")


def regenerate_runcommands(rundir: str, from_cached: bool = False) -> None:  # noqa: FBT001, FBT002
    """Regenerate runcommands for missing or truncated runs.

    Args:
        rundir (str): Path to the rundir.
        from_cached (bool, optional): Load experiment status data from 'runstatus.csv'. Defaults to False.
    """
    if from_cached:
        logger.info("Loading experiment status data from 'runstatus.csv'...")
        data = pd.read_csv("runstatus.csv")
        logger.info("Done!")
    else:
        logger.info("Scanning rundirs for experiment status...")
        data = check_missing(rundir=rundir)
        logger.info("Done!")

    if len(data) > 0:
        generate_commands(data, RunStatus.MISSING, rundir)
        generate_commands(data, RunStatus.TRUNCATED, rundir)
    else:
        logger.info(f"But nothing found at {rundir}.")


if __name__ == "__main__":
    fire.Fire(regenerate_runcommands)
