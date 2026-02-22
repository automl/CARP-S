"""Gather data and check for missing/truncated runs in one pass."""

from __future__ import annotations

import os
import json
import fire
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from omegaconf import OmegaConf

# Import existing carps utilities based on provided source logic
from carps.analysis.gather_data_utils import (
    load_log, 
    normalize_logs, 
    convert_mixed_types_to_str,
    read_jsonl_content
)
from carps.utils.check_missing import generate_commands
from carps.utils.loggingutils import get_logger, setup_logging
from carps.utils.types import RunStatus

setup_logging()
logger = get_logger(__file__)

def get_run_info(config_path: Path, log_fn: str = "trial_logs.jsonl") -> dict:
    """
    Combined worker function: Determines the execution status of a run and loads its log data.

    This function serves as the core processing unit for a single experiment directory. 
    It identifies whether a run is Completed, Truncated, or Missing based on the 
    expected number of trials in the config versus the actual trials in the logs.

    Args:
        config_path (Path): Path to the hydra 'config.yaml' file for a specific run.
        log_fn (str): The filename of the trial logs. Defaults to "trial_logs.jsonl".

    Returns:
        dict: A dictionary containing:
            - "status_info": Metadata including RunStatus and hydra overrides.
            - "log_df": A DataFrame of the processed trial logs.
            - "cfg_fn": String path to the configuration file.
            - "cfg_str": String representation of the configuration for serialization.
            Returns an empty dict if the config is invalid or lacks task resources.
    """
    rundir = config_path.parent.parent
    status = RunStatus.MISSING
    log_df = pd.DataFrame()
    
    # 1. Load Config
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        logger.error(f"Could not load config at {config_path}: {e}")
        return {}

    if not hasattr(cfg, "task") or not hasattr(cfg.task, "optimization_resources"):
        return {}

    # 2. Determine Status (Logic from check_missing.py)
    n_trials = cfg.task.optimization_resources.n_trials
    trial_logs_fn = rundir / log_fn
    
    if trial_logs_fn.is_file():
        try:
            # Check trial counts to determine if run finished
            trial_logs = read_jsonl_content(str(trial_logs_fn))
            if not trial_logs.empty and "n_trials" in trial_logs:
                n_trials_done = trial_logs["n_trials"].max()
                status = RunStatus.COMPLETED if n_trials_done >= n_trials else RunStatus.TRUNCATED
            
            # 3. Load and Process Log Data (Logic from gather_data.py)
            log_df = load_log(rundir, log_fn=log_fn)
        except Exception as e:
            logger.warning(f"Error processing logs in {rundir}: {e}")

    # 4. Extract Overrides for command generation
    try:
        hydra_cfg = OmegaConf.load(config_path.parent / "hydra.yaml")
        task_overrides = hydra_cfg.hydra.overrides.task
        hydra_overrides = hydra_cfg.hydra.overrides.hydra
    except Exception:
        task_overrides = []
        hydra_overrides = []

    status_info = {
        "status": status.name,
        "benchmark_id": getattr(cfg, "benchmark_id", "unknown"),
        "task_id": getattr(cfg, "task_id", "unknown"),
        "optimizer_id": getattr(cfg, "optimizer_id", "unknown"),
        "seed": getattr(cfg, "seed", -1),
        "task_overrides": " ".join(task_overrides),
        "hydra_overrides": " ".join(hydra_overrides),
    }

    return {
        "status_info": status_info,
        "log_df": log_df,
        "cfg_fn": str(config_path),
        "cfg_str": OmegaConf.to_yaml(cfg).replace("\n", "\\n")
    }

def gather_and_check(
    rundir: str | list[str],
    log_fn: str = "trial_logs.jsonl",
    n_processes: int | None = None,
    outdir: str | Path | None = None
) -> None:
    """
    Scans directories to gather performance logs and check for missing/truncated runs.

    This is the main entry point. It performs a parallel scan of the provided directories, 
    generates a status report (`runstatus.csv`), creates shell scripts to restart failed 
    runs (`runcommands_*.sh`), and aggregates all valid trial data into consolidated 
    CSV and Parquet files.

    Args:
        rundir (str | list[str]): One or more directories to scan for results.
        log_fn (str): The filename of the trial logs. Defaults to "trial_logs.jsonl".
        n_processes (int | None): Number of CPU processes for parallel processing. 
            Defaults to None (uses all available cores).
        outdir (str | Path | None): Directory where output files will be saved. 
            If None, uses the common path of input rundirs.

    Returns:
        None: Outputs files directly to the file system (logs.csv, runstatus.csv, etc.).
    """
    if isinstance(rundir, str):
        rundir = [rundir]
    
    all_status_data = []
    all_log_dfs = []
    config_mappings = []
    
    for r in rundir:
        logger.info(f"Scanning {r} for experiment configs...")
        # Find every experiment directory via its hydra config
        config_paths = list(Path(r).glob("**/.hydra/config.yaml"))
        logger.info(f"Found {len(config_paths)} experiment directories.")

        worker = partial(get_run_info, log_fn=log_fn)
        with Pool(processes=n_processes) as pool:
            results = pool.map(worker, config_paths)

        for res in results:
            if not res: continue
            all_status_data.append(res["status_info"])
            if not res["log_df"].empty:
                # Store log and track config for cfg_str/cfg_fn mapping
                all_log_dfs.append(res["log_df"])
                config_mappings.append({"cfg_fn": res["cfg_fn"], "cfg_str": res["cfg_str"]})

    # --- PART 1: Handle Status and Run-Commands ---
    status_df = pd.DataFrame(all_status_data).dropna()
    if outdir is None:
        outdir = Path(os.path.commonpath(rundir))
    else:
        outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    status_df.to_csv(outdir / "runstatus.csv", index=False)
    logger.info(f"Saved run status to {outdir / 'runstatus.csv'}")
    
    # Generate shell scripts to fix non-completed runs
    generate_commands(status_df, RunStatus.MISSING, str(outdir))
    generate_commands(status_df, RunStatus.TRUNCATED, str(outdir))

    # --- PART 2: Handle Data Gathering ---
    if all_log_dfs:
        logger.info("Consolidating and normalizing logs...")
        df = pd.concat(all_log_dfs).reset_index(drop=True)
        
        # Create metadata mapping between experiments and their config strings
        df_cfg = pd.DataFrame(config_mappings).drop_duplicates()
        df_cfg["experiment_id"] = np.arange(len(df_cfg))
        
        # Assign experiment_id back to main log dataframe
        mapping = dict(zip(df_cfg["cfg_fn"], df_cfg["experiment_id"]))
        df["experiment_id"] = df["cfg_fn"].map(mapping)
        
        # Apply normalization and cleanup
        df = normalize_logs(df)
        df = convert_mixed_types_to_str(df)
        df_cfg = convert_mixed_types_to_str(df_cfg)

        # Save aggregated outputs
        df.to_csv(outdir / "logs.csv", index=False)
        df.to_parquet(outdir / "logs.parquet", index=False)
        df_cfg.to_csv(outdir / "logs_cfg.csv", index=False)
        df_cfg.to_parquet(outdir / "logs_cfg.parquet", index=False)
        
        logger.info(f"Gathered logs for {len(all_log_dfs)} runs into {outdir}")
    else:
        logger.warning("No log data found to gather.")

    logger.info("Done! 😊")

if __name__ == "__main__":
    fire.Fire(gather_and_check)