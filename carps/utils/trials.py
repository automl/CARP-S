from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from ConfigSpace import Configuration
from dataclasses_json import dataclass_json


class StatusType(IntEnum):
    """Class to define status types of configs."""

    RUNNING = 0  # In case a job was submitted, but it has not finished.
    SUCCESS = 1
    CRASHED = 2
    TIMEOUT = 3
    MEMORYOUT = 4


@dataclass_json
@dataclass(frozen=True)
class TrialInfo:
    """Information about a trial.

    Parameters
    ----------
    config : Configuration
    instance : int | None, defaults to None, length 
    seed : int | None, defaults to None
    budget : float | None, defaults to None
    name: str | None, defaults to None, arbitrary information, length 100
    checkpoint: str | None, defaults to None, checkpoint path, length 250

    The length of the strings depends on the setting for the database.
    """

    config: Configuration
    instance: int | None = None
    seed: int | None = None
    budget: float | None = None
    name: str | None = None
    checkpoint: str | None = None


@dataclass_json
@dataclass(frozen=True)
class TrialValue:
    """Values of a trial.

    Parameters
    ----------
    cost : float | list[float]
    time : float, defaults to 0.0
    virtual_time : float, defaults to 0.0
    status : StatusType, defaults to StatusType.SUCCESS
    starttime : float, defaults to 0.0
    endtime : float, defaults to 0.0
    additional_info : dict[str, Any], defaults to {}
    """

    cost: float | list[float]
    time: float = 0.0
    virtual_time: float = 0.0
    status: StatusType = StatusType.SUCCESS
    starttime: float = 0.0
    endtime: float = 0.0
    additional_info: dict[str, Any] = field(default_factory=dict)
