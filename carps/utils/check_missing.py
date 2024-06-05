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
    COMPLETED = auto()
    MISSING = auto()
    TRUNCATED = auto()


def get_experiment_status(path: Path) -> dict:
    status = RunStatus.MISSING

    cfg = OmegaConf.load(path)
    n_trials = cfg.task.n_trials
    trial_logs_fn = path.parent.parent / "trial_logs.jsonl"
    if trial_logs_fn.is_file():
        trial_logs = read_jsonl_content(str(trial_logs_fn))
        n_trials_done = trial_logs["n_trials"].max()
        status = RunStatus.COMPLETED if n_trials == n_trials_done else RunStatus.TRUNCATED

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


def check_missing(rundir: str, n_processes: int = 4) -> pd.DataFrame:
    rundir = Path(rundir)
    paths = rundir.glob("**/.hydra/config.yaml")
    with Pool(processes=n_processes) as pool:
        data = pool.map(get_experiment_status, paths)
    data = pd.DataFrame(data)
    data.to_csv("runstatus.csv", index=False)
    return data


def generate_commands(missing_data: pd.DataFrame, runstatus: RunStatus, rundir: str = "") -> None:
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


def regenerate_runcommands(rundir: str, from_cached: bool = False) -> None:
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
