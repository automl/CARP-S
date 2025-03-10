"""Run optimizer, make task and optimizer."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import (
    inspect,
    print as printr,
)

from carps.utils.exceptions import NotSupportedError

if TYPE_CHECKING:
    from py_experimenter.result_processor import ResultProcessor

    from carps.optimizers.optimizer import Optimizer
    from carps.utils.task import Task


def make_task(cfg: DictConfig, result_processor: ResultProcessor | None = None) -> Task:
    """Make ObjectiveFunction.

    Parameters
    ----------
    cfg : DictConfig
        Global configuration.
    result_processor : ResultProcessor
        Py experimenter result processor, important for logging. Necessary to
        instantiate database logger.

    Returns:
    -------
    Task
        Task (objective function with defined input and output space) to optimize.
    """
    task_cfg = cfg.task

    loggers = []
    if "loggers" in cfg:
        for logger in cfg.loggers:
            if "DatabaseLogger" in logger._target_:
                kwargs = {"result_processor": result_processor}
            elif "FileLogger" in logger._target_:
                kwargs = {"directory": cfg.outdir}
            else:
                kwargs = {}
            logger_instantiated = instantiate(logger)(**kwargs)
            loggers.append(logger_instantiated)
    return instantiate(task_cfg, objective_function={"loggers": loggers})


def make_optimizer(cfg: DictConfig, task: Task) -> Optimizer:
    """Make Optimizer.

    Parameters
    ----------
    loggers : list[AbstractLogger]
        List of loggers to use.
    cfg : DictConfig
        Global configuration.
    task : Task
        Task to optimize.

    Returns:
    -------
    Optimizer
        Instantiated optimizer.
    """
    optimizer_cfg = cfg.optimizer
    optimizer = instantiate(optimizer_cfg)(task=task, loggers=task.objective_function.loggers)
    if "optimizer_wrappers" in cfg:
        for wrapper in cfg.optimizer_wrappers:
            optimizer = wrapper(optimizer)
    return optimizer


def optimize(cfg: DictConfig, result_processor: ResultProcessor | None = None) -> None:
    """Run optimizer on task.

    Save trajectory and metadata to database.

    Parameters
    ----------
    cfg : DictConfig
        Global configuration.
    result_processor : ResultProcessor
        Py experimenter result processor, important for logging. Necessary to
        instantiate database logger.

    """
    os.environ["HYDRA_FULL_ERROR"] = "1"

    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
    printr(cfg_dict)
    # hydra_cfg = HydraConfig.instance().get()
    # printr(hydra_cfg.run.dir)

    task = make_task(cfg=cfg, result_processor=result_processor)
    inspect(task)

    optimizer = make_optimizer(cfg=cfg, task=task)
    inspect(optimizer)

    try:
        inc_tuple = optimizer.run()
        printr("Solution found: ", inc_tuple)
    except NotSupportedError:
        print("Not supported. Skipping.")
    except Exception as e:
        print("Something went wrong:")
        print(e)
        raise e
