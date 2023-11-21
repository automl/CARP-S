import typing

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from ConfigSpace import Configuration
from hpbandster.optimizers import BOHB as BOHB

from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.optimizers.optimizer import Optimizer
from smacbenchmarking.utils.trials import TrialInfo


class BOHBOptimizer(Optimizer):
    def __init__(self, problem: Problem, n_iterations, bohb: BOHB, nameserver: hpns.NameServer, run_id, host) -> None:
        """
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
        self.bohb = bohb(self.configspace, run_id=run_id)
        self.nameserver = nameserver
        self.host = host
        self.trajectory = []

    def convert_to_trial(self, config: Configuration) -> TrialInfo:
        """Convert proposal from BOHB to TrialInfo."""
        return TrialInfo(config=config)

    def get_trajectory(self, sort_by: str = "trials") -> typing.Tuple[typing.List[float], typing.list[float]]:
        """Get trajectory of optimizer."""
        return list(range(len(self.trajectory))), self.trajectory  # FIXME: Check trajectory!

    def setup_bohb(self, cfg) -> None:
        """Setup BOHB optimizer."""
        # FIXME: interact with the problem or wrap it to meet the desired interface of the worker

        class myWorker:
            def __init__(self, nameserver, run_id) -> None:
                self.nameserver = nameserver
                self.run_id = run_id

            def run(self, background=False) -> None:
                pass

            def compute(self, config, budget, **kwargs) -> float:
                pass

        self.w = myWorker(nameserver=cfg.host, run_id=cfg.run_id)

    def run(self) -> None:
        """Run optimizer."""
        self.nameserver.start()

        self.w.run(background=True)

        # TODO analyse the return trajectory.
        res = self.bohb.run(n_iterations=self.n_iterations)
        res
        hpres  # TODO what to use this for?

        self.bohb.shutdown(shutdown_workers=True)
        self.nameserver.shutdown()

        # timeout = self.cfg.timeout
        # start_time = time()
        # while time() - start_time < timeout:
        #     sleep(1)
        #     trial = self.problem.configspace.sample_configuration()
        #     trial = self.convert_to_trial(trial)
        #     result = self.problem.evaluate(trial)
        #     self.trajectory.append(result.cost)
        return
