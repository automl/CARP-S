from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pandas as pd


def get_color_palette(df: pd.DataFrame | None) -> dict[str, Any]:
    cmap = sns.color_palette("colorblind", as_cmap=True)
    optimizers = list(df["optimizer_id"].unique())
    optimizers.sort()
    return {o: cmap[i] for i, o in enumerate(optimizers)}


def savefig(fig: plt.Figure, filename: str) -> None:
    figure_filename = Path(filename)
    figure_filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(figure_filename) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(figure_filename) + ".pdf", dpi=300, bbox_inches="tight")


def setup_seaborn(font_scale: float | None = None) -> None:
    if font_scale is not None:
        sns.set_theme(font_scale=font_scale)
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")


def filter_only_final_performance(
    df: pd.DataFrame, budget_var: str = "n_trials_norm", max_budget: float = 1, soft: bool = True
) -> pd.DataFrame:
    if not soft:
        df = df[np.isclose(df[budget_var], max_budget)]
    else:
        df = df[df.groupby(["optimizer_id", "problem_id", "seed"])[budget_var].transform(lambda x: x == x.max())]
    return df
