"""Calculate runtime in CPU hours per task type from the log file."""
from __future__ import annotations

import fire
import pandas as pd
from carps.analysis.utils import filter_only_final_performance


def calc(x: pd.DataFrame) -> float:
    n_optimizers = x["optimizer_id"].nunique()
    t = x["time"].sum() / n_optimizers
    return t / 3600

def main(log_fn: str = "results/subsets/logs.parquet") -> None:
    """Calculate the runtime in CPU hours per task type from the log file.

    Args:
        log_fn (str): Path to the log file in Parquet format.
    """
    df = pd.read_parquet(log_fn)  # noqa: PD901
    df_final = filter_only_final_performance(df=df, x_column="n_trials")

    runtime_df = df_final.groupby(by=["task_type"]).apply(calc)
    runtime_df.name = "time"
    print(runtime_df)
    runtime_df.to_csv("runtimes.csv")


    df_rt = pd.read_csv("runtimes.csv", index_col="task_type").map(int)
    total = df_rt.sum()

    total = pd.DataFrame(total).T
    total.index = ["total"]

    df_rt = pd.concat([df_rt, total], axis=0)

    latex_str = df_rt.to_latex(
        buf="runtimes.tex", caption="Runtimes in CPU Hours per Task Type", label="tab:runtimes", index_names=True
    )
    print(latex_str)


if __name__ == "__main__":
    fire.Fire(main)