from __future__ import annotations

from pathlib import Path
import pandas as pd

import carps

from carps.analysis.run_autorank import get_df_crit
from carps.analysis.gather_data import convert_mixed_types_to_str


if __name__=="__main__":
    paths = {
        # "MF": [
        #     "runs/DEHB",
        #     "runs/SMAC3-Hyperband",
        #     "runs/SMAC3-MultiFidelityFacade",
        #     ],
        # "BB": [
        #     "runs/SMAC3-BlackBoxFacade",
        #     "runs/RandomSearch",
        #     "runs/Nevergrad-CMA-ES",
        # ],
        "BBsubset": [
            "runs_subset/SMAC3-BlackBoxFacade",
            "runs_subset/SMAC3-1.4-BlackBoxFacade",
            "runs_subset/SMAC3-HPOFacade",
            "runs_subset/RandomSearch",
            "runs_subset/Optuna",
            "runs_subset/HEBO",
            "runs_subset/Scikit_Optimize",
            "runs_subset/SyneTune-BO",
            "runs_subset/Nevergrad-CMA-ES",
        ],

    }
    always_convert = False

    normalize_peformance = True

    _log_fn_pq = "logs.parquet"
    _log_cfg_fn_pq = "logs_cfg.parquet"
    _log_fn_csv = "logs.csv"
    _log_cfg_fn_csv = "logs_cfg.csv"
    combined_fn = "logs_combined.parquet"
    combined_cfg_fn = "logs_combined_cfg.parquet"
    df_crit_fn = "pointfile.txt"

    def convert(fn_pq: Path, fn_csv: Path, always_convert: bool = True) -> None:
        if fn_csv.is_file() and not fn_pq.is_file() or always_convert:
            print("Convert", fn_csv, "to", fn_pq)
            df = pd.read_csv(fn_csv)
            df = convert_mixed_types_to_str(df)
            df.to_parquet(fn_pq)  

    for scenario, _paths in paths.items():
        for p in _paths:
            log_fn_pq = Path(p) / _log_fn_pq    
            log_cfg_fn_pq = Path(p) / _log_cfg_fn_pq  
            log_fn_csv = Path(p) / _log_fn_csv    
            log_cfg_fn_csv = Path(p) / _log_cfg_fn_csv

            convert(log_fn_pq, log_fn_csv, always_convert)
            convert(log_cfg_fn_pq, log_cfg_fn_csv, always_convert)

            


