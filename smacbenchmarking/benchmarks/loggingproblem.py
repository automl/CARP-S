from smacbenchmarking.benchmarks.problem import Problem
from abc import ABC, abstractmethod
from smacbenchmarking.loggers.abstract_logger import AbstractLogger
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class LoggingProblem(Problem, ABC):

    def __init__(self) -> None:
        super().__init__()
        self.loggers = list()

    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate problem.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns
        -------
        TrialValue
            Value of the trial, i.e.:
                - cost : float | list[float]
                - time : float, defaults to 0.0
                - status : StatusType, defaults to StatusType.SUCCESS
                - starttime : float, defaults to 0.0
                - endtime : float, defaults to 0.0
                - additional_info : dict[str, Any], defaults to {}
        """
        trial_value = self._evaluate(trial_info)
        for logger in self.loggers:
            logger.log_trial(trial_info, trial_value)
        return trial_value


    def add_logger(self, logger: AbstractLogger):
        """Add the given logger to the problem.

        Parameters
        ----------
        logger : AbstractLogger
            Logger that should be used for logging.
        """
        self.loggers.append(logger)

    @abstractmethod
    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate problem.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns
        -------
        TrialValue
            Value of the trial, i.e.:
                - cost : float | list[float]
                - time : float, defaults to 0.0
                - status : StatusType, defaults to StatusType.SUCCESS
                - starttime : float, defaults to 0.0
                - endtime : float, defaults to 0.0
                - additional_info : dict[str, Any], defaults to {}
        """
        raise NotImplementedError

