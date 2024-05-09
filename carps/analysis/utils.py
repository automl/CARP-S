from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_color_palette(df: pd.DataFrame | None) -> dict[str, Any]:
    cmap = sns.color_palette("colorblind", as_cmap=True)
    optimizers = list(df["optimizer_id"].unique())
    optimizers.sort()
    palette = {o: cmap[i] for i,o in enumerate(optimizers)}
    return palette

def savefig(fig: plt.Figure, filename: str) -> None:
    figure_filename = Path(filename)
    figure_filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_filename, dpi=300, bbox_inches="tight")

def setup_seaborn(font_scale: float | None = None) -> None:
    if font_scale is not None:
        sns.set_theme(font_scale=font_scale)
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")