"""Logging utilities for the project."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rich.logging import RichHandler


def setup_logging() -> None:
    """Setup logging module."""
    FORMAT = "%(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])


def get_logger(logger_name: str) -> logging.Logger:
    """Get the logger by name.

    Parameters
    ----------
    logger_name : str
        Name of the logger.

    Returns:
    --------
    logging.Logger
        Logger object.
    """
    setup_logging()
    return logging.getLogger(logger_name)


class CustomEncoder(json.JSONEncoder):
    """- Serializes python/Numpy objects via customizing json encoder.
    - **Usage**
        - `json.dumps(python_dict, cls=EncodeFromNumpy)` to get json string.
        - `json.dump(*args, cls=EncodeFromNumpy)` to create a file.json.
    """

    def default(self, obj: Any) -> Any:
        """Converts numpy objects to pure python objects.

        Parameters
        ----------
        obj : Any
            Object to be converted.

        Returns:
        --------
        Any
            Pure python object.
        """
        if isinstance(obj, np.int64 | np.int32):
            return int(obj)
        if isinstance(obj, np.float64 | np.float32):
            return float(obj)
        return super().default(obj)


def log_pip_freeze(file_path: str | Path) -> None:
    """Write the output of `pip freeze` directly to a file."""
    logger = get_logger("carps.utils.loggingutils.log_pip_freeze")
    try:
        # TODO: enable discovery and usage of uv
        result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)  # noqa: S603, S607
        with open(file_path, "a") as f:
            f.write("Installed packages (pip freeze):\n")
            f.write(result.stdout + "\n")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("Failed to run pip freeze. Error: %s", e)
        with open(file_path, "a") as f:
            f.write("Failed to run pip freeze:\n")
            f.write(str(e) + "\n")


def log_python_env(log_file: str | Path = "env_log.txt") -> None:
    """Log the Python environment details directly to a file."""
    with open(log_file, "w") as f:
        f.write("Python Environment Information\n")
        f.write("=" * 32 + "\n")
        f.write(f"Python Version: {sys.version}\n")
        f.write(f"Python Executable: {sys.executable}\n")
    log_pip_freeze(log_file)
