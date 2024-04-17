from __future__ import annotations

import logging
from rich.logging import RichHandler


def setup_logging() -> None:
    FORMAT = "%(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])