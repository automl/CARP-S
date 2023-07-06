from ConfigSpace import ConfigurationSpace
from omegaconf import DictConfig
from smac.runhistory import TrialInfo

from smacbenchmarking.optimizers.optimizer import Optimizer, SearchSpace


class ContainerizedOptimizer(Optimizer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(None)
        self._optimizer = None
        self._cfg = cfg

    def convert_to_trial(self, *args: tuple, **kwargs: dict) -> TrialInfo:
        return self._optimizer.convert_to_trial(*args, **kwargs)

    def get_trajectory(self, sort_by: str = "trials") -> tuple[list[float], list[float]]:
        return self._optimizer.get_trajectory(sort_by)

    def run(self) -> None:
        # get search space from problem

        # instantiate optimizer with
        pass

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        self._optimizer.convert_configspace(configspace)
