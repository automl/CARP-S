from abc import abstractmethod, ABC
from typing import Any
from typing_extensions import TypeAlias
from ..benchmarks.problem import Problem
from ConfigSpace import ConfigurationSpace
from smac.runhistory.dataclasses import TrialInfo


SearchSpace = TypeAlias(Any)

class Optimizer(ABC):
    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        super().__init__()

    @abstractmethod
    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        """Convert ConfigSpace configuration space to search space from optimizer.

        Parameters
        ----------
        configspace : ConfigurationSpace
            Configuration space.

        Returns
        -------
        SearchSpace
            Optimizer's search space.
        """
        raise NotImplementedError
    
    @abstractmethod
    def convert_to_trial(self, *args, **kwargs) -> TrialInfo:
        """Convert proposal by optimizer to TrialInfo.

        This ensures that the problem can be evaluated with a unified API.

        Returns
        -------
        TrialInfo
            Trial info containing configuration, budget, seed, instance.
        """
        raise NotImplementedError