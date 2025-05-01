"""Logging utilities for the project."""

from __future__ import annotations

import json
import logging
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
