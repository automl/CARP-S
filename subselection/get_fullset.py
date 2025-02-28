from __future__ import annotations

from pathlib import Path

import fire
import pandas as pd
from carps.analysis.gather_data import normalize_logs
from carps.analysis.run_autorank import get_df_crit


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
        rundir: str,
        optimizer_ids: list[str],
        output_dir: str = ".",
        normalize_performance: bool = True,

) -> pd.DataFrame:
    if len(optimizer_ids) != 3:
        raise ValueError(f"Please select only three optimizers for the benchmark subselection. Current selection: {optimizer_ids}."\
                         "The subselection can be in principle also done with more optimizers but this is currently not "\
                            "integrated nor tested.")

    df, df_cfg = load_set([rundir], set_id="full")

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
    fire.Fire(get_fullset)