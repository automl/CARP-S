from __future__ import annotations

from pathlib import Path
import pandas as pd

import carps

from carps.analysis.run_autorank import get_df_crit
from carps.analysis.process_data import process_logs


if __name__=="__main__":
    paths = {
    "MF": [
        "runs/SMAC3-Hyperband",
        "runs/SMAC3-MutliFidelityFacade",
        "runs/RandomSearch",
        ],
        "BB": [
            "runs/SMAC3-BlackBoxFacade",
            "runs/RandomSearch",
            "runs/Nevergrad-CMA-ES",
        ],

    }

    normalize_peformance = True

    _log_fn_pq = "logs.parquet"
    _log_cfg_fn_pq = "logs_cfg.parquet"
    _log_fn_csv = "logs.csv"
    _log_cfg_fn_csv = "logs_cfg.csv"
    combined_fn = "logs_combined.parquet"
    combined_cfg_fn = "logs_combined_cfg.parquet"
    df_crit_fn = "pointfile.txt"

    def convert(fn_pq: Path, fn_csv: Path) -> None:
        if fn_csv.is_file() and not fn_pq.is_file():
            print("Convert", fn_csv, "to", fn_pq)
            df = pd.read_csv(fn_csv)
            df.to_parquet(fn_pq)  

    for scenario, _paths in paths.items():
        for p in _paths:
            log_fn_pq = Path(p) / _log_fn_pq    
            log_cfg_fn_pq = Path(p) / _log_cfg_fn_pq  
            log_fn_csv = Path(p) / _log_fn_csv    
            log_cfg_fn_csv = Path(p) / _log_cfg_fn_csv

            convert(log_fn_pq, log_fn_csv)
            convert(log_cfg_fn_pq, log_cfg_fn_csv)

            


