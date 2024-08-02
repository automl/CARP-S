from __future__ import annotations

import fire
import pandas as pd
from pathlib import Path
from carps.analysis.run_autorank import get_df_crit
from carps.analysis.gather_data import normalize_logs, get_interpolated_performance_df, load_logs, process_logs


def load_set(paths: list[str], set_id: str = "unknown") -> tuple[pd.DataFrame, pd.DataFrame]:
    logs = []
    for p in paths:
        fn = Path(p) / "trajectory.parquet"
        if not fn.is_file():
            fn = Path(p) / "logs.parquet"
        logs.append(pd.read_parquet(fn))

    df = pd.concat(logs).reset_index(drop=True)
    df_cfg = pd.concat([pd.read_parquet(Path(p) / "logs_cfg.parquet") for p in paths]).reset_index(drop=True)
    df["set"] = set_id
    return df, df_cfg

def get_fullset(
        rundir: str | list[str], 
        optimizer_ids: list[str],
        output_dir: str = ".",
        benchmark_exclusions: list[str] | None = None,
        normalize_performance: bool = True,
        
) -> pd.DataFrame:
    if len(optimizer_ids) != 3:
        raise ValueError(f"Please select only three optimizers for the benchmark subselection. Current selection: {optimizer_ids}."\
                         "The subselection can be in principle also done with more optimizers but this is currently not "\
                            "integrated nor tested.")

    if not isinstance(rundir, list):
        rundir = [rundir]
    df, df_cfg = load_set(rundir, set_id="full")

    if benchmark_exclusions is not None:
        df = df[~df["benchmark_id"].isin(benchmark_exclusions)]

    df = normalize_logs(df)

    perf_col = "trial_value__cost_inc_norm" if normalize_performance else "trial_value__cost_inc"

    df_crit = get_df_crit(df, perf_col=perf_col)
    # index: problem_id, columns: optimizer_ids

    filename = Path(output_dir) / "df_crit.csv"
    filename.parent.mkdir(exist_ok=True, parents=True)
    df_crit.loc[:,optimizer_ids].to_csv(filename)

    return df_crit


if __name__ == "__main__":
    # in the subselection dir
    # python get_fullset.py ../runs_MO '["RandomSearch","Optuna-MO","Nevergrad-DE"]' MO_0
    # python get_fullset.py ../runs_MOMF '["RandomSearch","SMAC3-MOMF-GP","Nevergrad-DE"]' MOMF_0/default
    # python subselection/get_fullset.py '["runs/RandomSearch","runs/SMAC3-BlackBoxFacade","runs/Nevergrad-CMA-ES"]' '["RandomSearch","SMAC3-BlackBoxFacade","Nevergrad-CMA-ES"]' subselection/data/BB/round2
    fire.Fire(get_fullset)