"""Environment variables for carps root and data directory."""

from __future__ import annotations

import os
from pathlib import Path

CARPS_ROOT = (Path(__file__).parent / ".." / "..").resolve()
CARPS_TASK_DATA_DIR = os.environ.get("CARPS_TASK_DATA_DIR", Path(CARPS_ROOT) / "task_data")
