"""Index all task and optimizer configs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import fire
import omegaconf
import pandas as pd
from omegaconf import OmegaConf
from platformdirs import user_cache_dir
from rich.progress import track

from carps.utils.loggingutils import get_logger

logger = get_logger("ConfigIndexer")

cache_path = Path(user_cache_dir("carps")) / "index.csv"

config_folder = Path(__file__).parent.parent / "configs"
config_folder_task = config_folder / "task"
config_folder_optimizer = config_folder / "optimizer"

PATH_KEY_ZIP = {
    config_folder_task: "task_id",
    config_folder_optimizer: "optimizer_id",
}


def index_configs(extra_task_paths: list[str] | None = None, extra_optimizer_paths: list[str] | None = None) -> None:
    """Index all task and optimizer configs.

    Create `index.csv` containing the config filename `config_fn` and the
    `task_id` or `optimizer_id` for all task and optimizer configs.

    Parameters
    ----------
    extra_task_paths : list[str], optional
        Extra paths to custom tasks, must be a folder containing only task configs.
    extra_optimizer_paths : list[str], optional
        Extra paths to custom optimizers, must be a folder containing only optimizer configs.
    """
    register_extra_paths(extra_task_paths, extra_optimizer_paths)

    index_list = []
    for path, key in PATH_KEY_ZIP.items():
        paths = list(path.glob("**/*.yaml"))

        table_list = []
        for fn in track(paths, total=len(paths), description=f"Gathering for {key}..."):
            cfg = OmegaConf.load(fn)
            try:
                value = cfg.get(key, None)
            except omegaconf.errors.InterpolationToMissingValueError:
                cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=False)
                value = cfg_dict.get(key, None)
            if value is None:
                continue
            table_list.append(
                {
                    "config_fn": str(fn),
                    key: value,
                }
            )
        table = pd.DataFrame(table_list)
        index_list.append(table)
    index_df = pd.concat(index_list)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(cache_path, index=False)


def create_table(key: str, paths: list[Path], target: Path) -> None:
    """Create index table."""
    table_list = []
    for p in paths:
        cfg = OmegaConf.load(p)
        value = cfg.get(key)
        table_list.append(
            {
                "config_fn": str(p),
                key: value,
            }
        )
    table = pd.DataFrame(table_list)
    table.to_csv(target, index=False)


def hash_inputs(paths: list[Path]) -> str:
    """Hash inputs so that index file can be cached."""
    hasher = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as f:
            while chunk := f.read(16 * 1024 * 1024):
                hasher.update(chunk)
    return hasher.hexdigest()


def register_extra_paths(extra_task_paths: list[str] | None, extra_optimizer_paths: list[str] | None) -> None:
    """Register extra task and optimizer paths.

    Parameters
    ----------
    extra_task_paths : list[str]
        Extra paths to custom tasks, must be a folder containing only task configs.
    extra_optimizer_paths : list[str]
        Extra paths to custom optimizers, must be a folder containing only optimizer configs.
    """
    if not extra_task_paths:
        extra_task_paths = []
    if not extra_optimizer_paths:
        extra_optimizer_paths = []

    for optimizer_path_str in extra_optimizer_paths:
        PATH_KEY_ZIP[Path(optimizer_path_str)] = "optimizer_id"
    for task_path_str in extra_task_paths:
        PATH_KEY_ZIP[Path(task_path_str)] = "task_id"


def get_index() -> pd.DataFrame:
    """Get index of carps compatible tasks and optimizers.

    Reads from the cache directory.

    Returns:
    -------
    pd.DataFrame
        The index with `task_id` and `optimizer_id` with the corresponding
        config filename.
    """
    if not cache_path.is_file():
        logger.info(
            f"Index file not found at {cache_path}. Reindex config. Attention! "
            "If you have configs in your package, manually run "
            "python -m carps.utils.index_configs --extra_task_paths=... "
            "--extra_optimizer_paths=..."
        )
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        index_configs()

    return pd.read_csv(cache_path)


if __name__ == "__main__":
    fire.Fire(index_configs)
