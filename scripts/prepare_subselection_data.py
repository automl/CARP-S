from __future__ import annotations

import contextlib
from pathlib import Path

import pandas as pd
from carps.analysis.gather_data import convert_mixed_types_to_str, normalize_logs
from carps.analysis.run_autorank import get_df_crit
from carps.utils.loggingutils import get_logger, setup_logging

setup_logging()
logger = get_logger(__file__)


if __name__=="__main__":
    paths = {
        "MF": [
            "runs/DEHB",
            "runs/SMAC3-MultiFidelityFacade",
            "runs/SMAC3-Hyperband",
            ],
        "BB": [
            "runs/SMAC3-BlackBoxFacade",
            "runs/RandomSearch",
            "runs/Nevergrad-CMA-ES",
        ],

    }

    normalize_peformance = True

    extension = ".parquet"
    logger.info(f"Use extension {extension}")

    _log_fn = "logs" + extension
    _log_cfg_fn = "logs_cfg" + extension
    combined_fn = "logs_combined" + extension
    combined_cfg_fn = "logs_combined_cfg" + extension
    df_crit_fn = "pointfile.txt"
    regather = True

    for scenario, _paths in paths.items():
        logger.info(scenario)
        identifier = scenario + "_"
        if regather:
            dfs = []
            dfs_cfg = []
            for p in _paths:
                p = Path(p)
                logger.info(f"Load {p.resolve()!s}")
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

            if extension == ".csv":
                df.to_csv(identifier + combined_fn)
                df_cfg.to_csv(identifier + combined_cfg_fn)
            else:
                df.to_parquet(identifier + combined_fn)
                df_cfg.to_parquet(identifier + combined_cfg_fn)
        elif extension == ".csv":
            df = pd.read_csv(identifier + combined_fn)
        else:
            df = pd.read_parquet(identifier + combined_fn)
        logger.info("Normalize logs...")
        df = normalize_logs(df)
        df = convert_mixed_types_to_str(df)
        logger.info("Done")
        perf_col = "trial_value__cost_inc_norm" if normalize_peformance else "trial_value__cost_inc"
        logger.info("Get final performance")
        df_crit = get_df_crit(df, perf_col=perf_col)
        df_crit.to_csv(identifier + df_crit_fn, sep=" ", index=False, header=False)
        df_crit.to_csv(identifier + "df_crit.csv")
        with contextlib.suppress(Exception):
            df_crit.to_parquet(identifier + "df_crit.parquet")

