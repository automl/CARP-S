"""Find overrides for tasks and optimizers based on their IDs."""

from __future__ import annotations

import os
from pathlib import Path

import fire
import pandas as pd

from carps.utils.index_configs import index_configs
from carps.utils.loggingutils import get_logger

logger = get_logger(__file__)

config_folder = Path(__file__).parent.parent / "configs"
config_folder_task = config_folder / "task"
config_folder_optimizer = config_folder / "optimizer"


def find_override(task_id: str | None = None, optimizer_id: str | None = None) -> str | None:
    """Find the override string for a task or optimizer based on its ID.

    Parameters
    ----------
    task_id : str, optional
        The ID of the task to find the override for.
    optimizer_id : str, optional
        The ID of the optimizer to find the override for.

    Returns:
    -------
    str
        The override string for the task or optimizer or None if nothing is found.
    """
    if task_id is not None:
        key = "task_id"
        path = config_folder_task
        to_find = task_id
    elif optimizer_id is not None:
        key = "optimizer_id"
        path = config_folder_optimizer
        to_find = optimizer_id
    else:
        raise ValueError("Please specify either `task_id` or `optimizer_id`.")

    index_fn = path / "index.csv"
    if not index_fn.is_file():
        index_configs()
    table = pd.read_csv(index_fn)

    try:
        config_fn = table["config_fn"][table[key] == to_find].to_numpy()[0]
        stripped_path = str(config_fn)[len(str(config_folder)) + 1 : -len(".yaml")]
        index = next(x for x, v in enumerate(stripped_path) if v == "/")
        return "+" + stripped_path[:index] + "=" + stripped_path[index + 1 :]
    except Exception as e:  # noqa: BLE001
        logger.info(f"Nothing found for {to_find} in config path {path}. Error: {e}")
        return None


def merge_overrides(overrides: list[str | None]) -> str:
    """Merge multiple overrides into one.

    If we can find a common path among the overrides, we merge them into one
    for a hydra grid.

    Parameters
    ----------
    overrides : list[str], optional
        The overrides to merge.

    Returns:
    -------
    str
        The merged override string.
    """
    # overrides = [
    #     "+task=YAHPO/MO/cfg_rbv2_super_1457",
    #     "+task=YAHPO/MO/cfg_rbv2_super_1452",
    #     "+task=YAHPO/MO/cfg_rbv2_super_1453",
    #     "+task=YAHPO/MO/cfg_rbv2_super_1454",
    # ]
    overrides_clean: list[str] = [o for o in overrides if o is not None]
    if len(overrides_clean) > 1:
        overrides_clean = list(map(str, overrides_clean))
        common = os.path.commonpath(overrides_clean)
        index = common.find("=")
        override = common[: index + 1] + ",".join([p[index + 1 :] for p in overrides_clean])
    else:
        override = overrides_clean[0]
    return override


if __name__ == "__main__":
    fire.Fire(find_override)
