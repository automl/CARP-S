from __future__ import annotations

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import inspect
from rich import print as printr

from smacbenchmarking.benchmarks.loggingproblemwrapper import LoggingProblemWrapper
from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.loggers.file_logger import FileLogger, dump_logs
from smacbenchmarking.optimizers.optimizer import Optimizer
from smacbenchmarking.utils.exceptions import NotSupportedError


def make_problem(cfg: DictConfig, logging: bool = False) -> Problem:
    """Make Problem

    Parameters
    ----------
    cfg : DictConfig
        Global configuration.
    logging : bool
        By default False, whether the problem should be wrapped in possibly
        specified loggers.

    Returns
    -------
    Problem
        Target problem.
    """
    problem_cfg = cfg.problem
    problem = instantiate(problem_cfg)

    if logging:
        # if logging and "loggers" in cfg and cfg.loggers is not None:
        #     for logger in cfg.loggers:
        #         loggercls = instantiate(logger)
        #         problem = loggercls(problem=problem, cfg=cfg)

        logging_problem_wrapper = LoggingProblemWrapper(problem=problem)

        # logging_problem_wrapper.add_logger(DatabaseLogger(result_processor))
        logging_problem_wrapper.add_logger(FileLogger())
        problem = logging_problem_wrapper

    return problem


def make_optimizer(cfg: DictConfig, problem: Problem) -> Optimizer:
    """Make Optimizer

    Parameters
    ----------
    cfg : DictConfig
        Global configuration
    problem : Problem
        Target problem

    Returns
    -------
    Optimizer
        Instantiated optimizer.
    """
    optimizer_cfg = cfg.optimizer
    optimizer = instantiate(optimizer_cfg)(problem=problem)
    return optimizer

def optimize(cfg: DictConfig) -> None:
    """Run optimizer on problem.

    Save trajectory and metadata to database.

    Parameters
    ----------
    cfg : DictConfig
        Global configuration.

    """
    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
    printr(cfg_dict)
    hydra_cfg = HydraConfig.instance().get()
    printr(hydra_cfg.run.dir)

    problem = make_problem(cfg=cfg, logging=True)
    inspect(problem)

    optimizer = make_optimizer(cfg=cfg, problem=problem)
    inspect(optimizer)

    try:
        optimizer.run()
    except NotSupportedError:
        print("Not supported. Skipping.")
    except Exception as e:
        print("Something went wrong:")
        print(e)
        raise e

    return None
