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


def save_run(cfg: DictConfig, optimizer: Optimizer, metadata: dict | None = None) -> None:
    """Save Run Data

    Save run to global database.

    Parameters
    ----------
    cfg : DictConfig
        Global configuration of run.
    optimizer : Optimizer
        Optimizer, needed to extract the trajectories.
    metadata : dict | None
        Optional metadata, by default None.
    """
    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
    # cfg_dict = pd.json_normalize(cfg_dict, sep=".").iloc[0].to_dict()  # flatten cfg

    trajectory_data = {}
    for sort_by in ["trials", "walltime"]:
        try:
            X, Y = optimizer.get_trajectory(sort_by=sort_by)
            trajectory_data[sort_by] = {"X": X, "Y": Y}
        except NotSupportedError:
            continue

    if metadata is None:
        metadata = {}

    data = {
        "cfg": cfg_dict,
        "metadata": metadata,
        "rundata": {
            "trajectory": trajectory_data,
        },
    }

    filename = "rundata.json"
    dump_logs(log_data=data, filename=filename)


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