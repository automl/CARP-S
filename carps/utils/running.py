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
from carps.benchmarks.wrapper import ParallelProblemWrapper
from benchmark_simulator import ObjectiveFuncWrapper
from carps.benchmarks.wrapper import OptimizerParallelWrapper

from functools import partial

if TYPE_CHECKING:
    from py_experimenter.result_processor import ResultProcessor

    from carps.benchmarks.problem import Problem
    from carps.optimizers.optimizer import Optimizer


def make_problem(cfg: DictConfig, result_processor: ResultProcessor | None = None) -> Problem:
    """Make Problem.

    Parameters
    ----------
    cfg : DictConfig
        Global configuration.
    result_processor : ResultProcessor
        Py experimenter result processor, important for logging. Necessary to
        instantiate database logger.

    Returns:
    -------
    Problem
        Target problem.
    """
    problem_cfg = cfg.problem
    loggers = []
    if "loggers" in cfg:
        for logger in cfg.loggers:
            if "DatabaseLogger" in logger._target_:
                kwargs = {"result_processor": result_processor}
            elif "FileLogger" in logger._target_:
                kwargs = {"directory": cfg.outdir}
            else:
                kwargs = {}
            logger = instantiate(logger)(**kwargs)
            loggers.append(logger)

    problem = instantiate(problem_cfg, loggers=loggers)
    if cfg.task.n_workers > 1:
        problem.evaluate = ParallelProblemWrapper(
            obj_func=problem.parallel_evaluate,
            obj_keys=[*list(cfg.task.objectives), "runtime"],
            fidel_keys=[cfg.task.fidelity_type] if cfg.task.fidelity_type else None,
            n_workers=cfg.task.n_workers,
            ask_and_tell=False
        )
    return problem


def make_optimizer(cfg: DictConfig, problem: Problem) -> Optimizer:
    """Make Optimizer.

    Parameters
    ----------
    loggers : list[AbstractLogger]
        List of loggers to use.
    cfg : DictConfig
        Global configuration
    problem : Problem
        Target problem

    Returns:
    -------
    Optimizer
        Instantiated optimizer.
    """
    optimizer_cfg = cfg.optimizer
    optimizer = instantiate(optimizer_cfg)(problem=problem, task=cfg.task, loggers=problem.loggers)
    if "optimizer_wrappers" in cfg:
        for wrapper in cfg.optimizer_wrappers:
            optimizer = wrapper(optimizer)
    return optimizer


def optimize(cfg: DictConfig, result_processor: ResultProcessor | None = None) -> None:
    """Run optimizer on problem.

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

    problem = make_problem(cfg=cfg, result_processor=result_processor)
    inspect(problem)

    if cfg.task.n_workers > 1:
        cfg_copy = cfg.copy()
        cfg_copy.task.n_workers = 1
        optimizer = make_optimizer(cfg=cfg_copy, problem=problem)
        inspect(optimizer)
        opt = OptimizerParallelWrapper(optimizer=optimizer)
        obj_fun = partial(problem.parallel_evaluate, obj_keys=optimizer.task.objectives)
        worker = ObjectiveFuncWrapper(
            save_dir_name="tmp",
            ask_and_tell=True,
            n_workers=cfg.task.n_workers,
            obj_func=obj_fun,
            n_actual_evals_in_opt=cfg.task.n_trials + cfg.task.n_workers,  # TODO check if trial for simulator means the same as in carps
            n_evals=cfg.task.n_trials,
            seed=cfg.seed,
            fidel_keys=None,
            obj_keys=optimizer.task.objectives,
            # allow_parallel_sampling=True,
            expensive_sampler=True
        )
        worker.simulate(opt)

    else:
        optimizer = make_optimizer(cfg=cfg, problem=problem)
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
