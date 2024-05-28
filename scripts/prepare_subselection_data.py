from __future__ import annotations

from pathlib import Path
import pandas as pd

import carps

from carps.analysis.run_autorank import get_df_crit
from carps.analysis.process_data import process_logs


if __name__=="__main__":
    paths = [
        "runs/SMAC3-BlackBoxFacade",
        "runs/RandomSearch",
        "runs/Nevergrad-CMA-ES",
    ]

    normalize_peformance = True

    _log_fn = "logs.csv"
    _log_cfg_fn = "logs_cfg.csv"
    combined_fn = "logs_combined.csv"
    combined_cfg_fn = "logs_combined_cfg.csv"
    df_crit_fn = "pointfile.txt"

    if False:
        dfs = []
        dfs_cfg = []
        for p in paths:
            print("Load", p)
            log_fn = Path(p) / _log_fn    
            log_cfg_fn = Path(p) / _log_cfg_fn    
            if not log_fn.is_file() or not log_cfg_fn.is_file():
                raise ValueError(f"Run `python -m carps.analysis.gather_data {p}` to create log file.")
            dfs.append(pd.read_csv(log_fn))
            dfs_cfg.append(pd.read_csv(log_cfg_fn))
        df = pd.concat(dfs)
        df_cfg = pd.concat(dfs_cfg)
        del dfs
        del dfs_cfg
        df.to_csv(combined_fn)
        df_cfg.to_csv(combined_cfg_fn)

    df = pd.read_csv(combined_fn)
    df = process_logs(df)
    perf_col = "trial_value__cost_inc_norm" if normalize_peformance else "trial_value__cost_inc"
    df_crit = get_df_crit(df, perf_col=perf_col)
    df_crit.to_csv(df_crit_fn, sep=" ", index=False, header=False)
    df_crit.to_csv("df_crit.csv")

