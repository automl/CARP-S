from __future__ import annotations

import logging
import numpy as np
import json

from rich.logging import RichHandler


def setup_logging() -> None:
    FORMAT = "%(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])


def get_logger(logger_name: str) -> logging.Logger:
    """Get the logger by name."""
    return logging.getLogger(logger_name)

class CustomEncoder(json.JSONEncoder):
    """
    - Serializes python/Numpy objects via customizing json encoder.
    - **Usage**
        - `json.dumps(python_dict, cls=EncodeFromNumpy)` to get json string.
        - `json.dump(*args, cls=EncodeFromNumpy)` to create a file.json.
    """
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        return super(CustomEncoder, self).default(obj)