"""MF-Prior-Bench Problem.

* HOW TO USE:
* ------------
* MF-Prior-Bench Documentation: https://automl.github.io/mf-prior-bench/latest/setup/
* 1. Install MF-Prior-Bench using `pip install mf-prior-bench`
* 2. Download the data for the respective benchmarks using
*    `python -m mfpbench download --benchmark <benchmark_name> --datadir <data_dir_path>`
*     where <benchmark_name> is one of ["pd1", "jahs"]
* 3. Install requirements from "./container_recipes/benchmarks/MFPBench/MFPBench_requirements.txt"
*    NOTE: JAHSBench is commented out in the requirements file due to compatibility issues
* 4. Test example 1 (smac20 multifidelity on PD1 imagenet_resnet_512 benchmark):
*    `python carps/run.py +optimizer/smac20=multifidelity +problem/MFPBench/pd1=imagenet_resnet_512
*     seed=1 task.n_trials=25 data_dir=<data_dir_path>`
*    Test example 2 (smac20 multifidelity on all available MFHartmann benchmarks):
*    `python carps/run.py +optimizer/smac20=multifidelity '+problem/MFPBench/mfh=glob(*)' 'seed=range(0, 10)' -m`
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import mfpbench

from carps.benchmarks.problem import Problem
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from carps.loggers.abstract_logger import AbstractLogger

benchmarks_names = ["pd1", "jahs", "mfh"]

benchmarks = [
    # PD1
    "lm1b_transformer_2048",
    "translatewmt_xformer_64",
    "cifar100_wideresnet_2048",
    "imagenet_resnet_512",
    # MFHartmann
    "mfh3",
    "mfh6",
    "mfh3_terrible",
    "mfh3_bad",
    "mfh3_moderate",
    "mfh3_good",
    "mfh6_terrible",
    "mfh6_bad",
    "mfh6_moderate",
    "mfh6_good",
    # JAHSBench
    "jahs",  # NOTE: Untested, compatibility issues with some package versions
]

C = TypeVar("C", bound=mfpbench.Config)


class MFPBenchProblem(Problem):
    """MF-Prior-Bench Problem class."""

    def __init__(
        self,
        benchmark_name: str,
        metric: str | list[str],
        benchmark: str | None = None,
        budget_type: str | None = None,
        prior: str | Path | C | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
        benchmark_kwargs: dict | None = None,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        """Initialize a MF-Prior-Bench problem."""
        super().__init__(loggers)

        self.benchmark_name = benchmark_name
        self.budget_type = budget_type
        self.benchmark = benchmark
        self.metrics = metric
        self.prior = prior
        self.perturb_prior = perturb_prior
        assert self.benchmark_name in benchmarks_names, f"benchmark_name must be one of {benchmarks_names}"
        assert self.benchmark in benchmarks, f"benchmark '{benchmark}' must be one of {benchmarks}"

        if benchmark_kwargs is None:
            benchmark_kwargs = {}
        elif benchmark_kwargs.get("datadir") is not None:
            # Assumes that the data is stored in the following format:
            benchmark_kwargs["datadir"] = Path(benchmark_kwargs["datadir"]) / benchmark_name

        self._problem = mfpbench.get(
            name=benchmark,
            prior=self.prior,
            perturb_prior=self.perturb_prior,
            **benchmark_kwargs,
        )
        self._configspace = self._problem.space

    @property
    def configspace(self) -> ConfigurationSpace:
        """Return configuration space.

        Returns:
        -------
        ConfigurationSpace
            Configuration space.
        """
        return self._configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate problem.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns:
        -------
        TrialValue
            Cost
        """
        configuration = trial_info.config
        start_time = time.time()
        result = self._problem.query(
            configuration.get_dictionary(),
            at=int(trial_info.budget) if trial_info.budget is not None else None,
        ).as_dict()
        end_time = time.time()

        ret = [result[metric] for metric in self.metrics]
        if len(ret) == 1:
            ret = ret[0]

        vt = 0.0
        if self.benchmark_name == "jahs":
            vt = result["runtime"]
        elif self.benchmark_name == "mfh":
            vt = result["fid_cost"]
        else:
            vt = result["train_cost"]

        return TrialValue(
            cost=ret,
            time=end_time - start_time,
            starttime=start_time,
            endtime=end_time,
            virtual_time=vt,
        )
