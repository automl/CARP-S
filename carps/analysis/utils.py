from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pandas as pd


colorblind_palette = ["#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499", "#DDDDDD"]


def get_color_palette(df: pd.DataFrame, model_name_key: str = "optimizer_id") -> dict[str, Any]:
    """Get a color palette based on the optimizers.

    Args:
        df (pd.DataFrame): Results dataframe.
        model_name_key (str, optional): The column name for the model name. Defaults to "model_name".

    Returns:
        dict[str, Any]: Color map.
    """
    optimizers = list(df[model_name_key].unique())
    optimizers.sort()
    cmap1 = colorblind_palette
    cmap2 = sns.color_palette("colorblind", as_cmap=False)
    cmap3 = sns.color_palette("Paired", as_cmap=False)
    colormaps = list(cmap1) + list(cmap2) + list(cmap3)
    assert len(optimizers) <= len(colormaps), f"Too many optimizers: {len(optimizers)} > {len(colormaps)}"
    return dict(zip(optimizers, colormaps, strict=False))


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
