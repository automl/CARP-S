from __future__ import annotations

import typing

import hpbandster.core.nameserver as hpns
from ConfigSpace import Configuration
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.optimizers.optimizer import Optimizer
from smacbenchmarking.utils.trials import TrialInfo


class BOHBOptimizer(Optimizer):
    def __init__(
            self, problem: Problem, n_iterations, bohb: BOHB, nameserver: hpns.NameServer, run_id,
            host
    ) -> None:
        """
        BOHB Optimizer.

        Wrapper build based on the following sequential example:
        https://automl.github.io/HpBandSter/build/html/auto_examples/example_1_local_sequential.html

        Parameters
        ----------
        problem : Problem (Benchmark)
            Problem to optimize.
        n_iterations : int
            Number of iterations to run.
        bohb : BOHB
            BOHB optimizer. (partially instantiated)
        nameserver : hpns.NameServer
            Nameserver.
        run_id : str
            Run ID. to distinguish workers and communicate with nameserver
        host : str
            Host. (IP address) to communicate between nameserver and workers
        """
        super().__init__(problem)
        self.configspace = self.problem.configspace
        self.n_iterations = n_iterations
        self.nameserver = nameserver
        self.host = host
        self.bohb = bohb  # partial
        self.run_id = run_id
        self.trajectory = []

    def convert_configspace(self, configspace: Configuration) -> typing.Any:
        """
        Convert ConfigSpace configuration space to search space from optimizer.

        Not required for BOHB, as it works directly with ConfigSpace.
        """
        pass

    def convert_to_trial(self, config: Configuration, budget) -> TrialInfo:
        """Convert proposal from BOHB to TrialInfo."""
        return TrialInfo(config=config, budget=budget)

    def setup_bohb(self) -> None:
        """Setup BOHB optimizer."""
        # FIXME: interact with the problem or wrap it to meet the desired interface of the worker
        self.nameserver.start()

        problem = self.problem
        convert_to_trial = self.convert_to_trial

        class MyWorker(Worker):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def compute(self, config, budget, **kwargs) -> dict:
                return {
                    "loss": float(problem.evaluate(convert_to_trial(config, budget))),
                    # this is the a mandatory field to run hyperband
                    "info": None,  # can be used for any user-defined information - also mandatory
                }

        self.w = MyWorker(nameserver=self.host, run_id=self.run_id)
        self.w.run(background=True)

        self.bohb = self.bohb(self.configspace, run_id=self.run_id)

    def teardown_bohb(self) -> None:
        """Teardown BOHB optimizer."""
        self.bohb.shutdown(shutdown_workers=True)
        self.nameserver.shutdown()

    def run(self) -> None:
        """Run optimizer."""
        self.setup_bohb()
        result = self.bohb.run(n_iterations=self.n_iterations)
        self.teardown_bohb()
