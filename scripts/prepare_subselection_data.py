from __future__ import annotations

from pathlib import Path
import pandas as pd

import carps

from carps.analysis.run_autorank import get_df_crit
from carps.analysis.process_data import process_logs


if __name__=="__main__":
    paths = {
    "MF": [
        "runs/DEHB",
        "runs/SMAC3-MultiFidelityFacade",
        "runs//SMAC3-Hyperband",
        ],
        "BB": [
            "runs/SMAC3-BlackBoxFacade",
            "runs/RandomSearch",
            "runs/Nevergrad-CMA-ES",
        ],

    }

    normalize_peformance = True
    extension = ".csv"

    _log_fn = "logs" + extension
    _log_cfg_fn = "logs_cfg" + extension
    combined_fn = "logs_combined" + extension
    combined_cfg_fn = "logs_combined_cfg" + extension
    df_crit_fn = "pointfile.txt"

    for scenario, _paths in paths.items():
        print(scenario)
        identifier = scenario + "_"
        if True:
            dfs = []
            dfs_cfg = []
            for p in _paths:
                p = Path(p)
                print("Load", p.resolve())
                log_fn = p / _log_fn    
                log_cfg_fn = p / _log_cfg_fn    
                if not log_fn.is_file() or not log_cfg_fn.is_file():
                    raise ValueError(f"Can't find {log_fn}. Run `python -m carps.analysis.gather_data {p}` to create log file.")
                
                if extension == ".csv":
                    df = pd.read_csv(log_fn)
                    df_cfg = pd.read_csv(log_cfg_fn)
                else:
                    df = pd.read_parquet(log_fn)
                    df_cfg = pd.read_parquet(log_cfg_fn)
                dfs.append(df)
                dfs_cfg.append(df_cfg)
            df = pd.concat(dfs)
            df_cfg = pd.concat(dfs_cfg)
            del dfs
            del dfs_cfg
            df.to_parquet(identifier + combined_fn)
            df_cfg.to_parquet(identifier + combined_cfg_fn)

        df = pd.read_parquet(identifier + combined_fn)
        perf_col = "trial_value__cost_inc_norm" if normalize_peformance else "trial_value__cost_inc"
        df_crit = get_df_crit(df, perf_col=perf_col)
        df_crit.to_csv(identifier + df_crit_fn, sep=" ", index=False, header=False)
        df_crit.to_parquet(identifier + "df_crit.parquet")

