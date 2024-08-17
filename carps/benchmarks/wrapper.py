from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmark_simulator import AbstractAskTellOptimizer, ObjectiveFuncWrapper
from ConfigSpace import Configuration

from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from carps.optimizers.optimizer import Optimizer


class ParallelProblemWrapper(ObjectiveFuncWrapper):
    def __call__(self, trial_info: TrialInfo) -> TrialValue:
        config = trial_info.config
        eval_config = dict(config)
        budget = trial_info.budget
        fidels = {self.fidel_keys[0]: budget} if budget else None
        print(">>>>>>>>", fidels)
        output = super().__call__(eval_config, fidels=fidels, trial_info=trial_info, obj_keys=self.obj_keys)
        print("<<<<<<<<<, done")

        time = None
        if "runtime" in self.obj_keys:
            time = output["runtime"]

        if len(self.obj_keys) > 1:
            cost = [output[k] for k in self.obj_keys if k != "runtime"]
        else:
            cost = output[self.obj_keys[0]]

        return TrialValue(cost=cost, time=time)


class OptimizerParallelWrapper(AbstractAskTellOptimizer):
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

        super().__init__()

        if self.optimizer.solver is None:
            self.optimizer.setup_optimizer()

        # we need to record the entire information preserved in the trial info during ask such that no information
        # is lost when we feed the information to the benchmark_simulator
        # NOTE: this solution does not solve the cases where one configuration runs on multiple seeds and instances!
        self.history: dict[Configuration, TrialInfo] = {}

    def ask(self) -> tuple[dict[str, Any], dict[str, int | float] | None, int | None]:
        """The ask method to sample a configuration using an optimizer.

        Args:
            None

        Returns:
            (eval_config, fidels) (tuple[dict[str, Any], dict[str, int | float] | None]):
                * eval_config (dict[str, Any]):
                    The configuration to evaluate.
                    The key is the hyperparameter name and its value is the corresponding hyperparameter value.
                    For example, when returning {"alpha": 0.1, "beta": 0.3}, the objective function evaluates
                    the hyperparameter configuration with alpha=0.1 and beta=0.3.
                * fidels (dict[str, int | float] | None):
                    The fidelity parameters to be used for the evaluation of the objective function.
                    If not multi-fidelity optimization, simply return None.
                * config_id (int | None):
                    The identifier of configuration if needed for continual learning.
                    Not used at all when continual_max_fidel=None.
                    As we internally use a hash of eval_config, it may be unstable if eval_config has float.
                    However, even if config_id is not provided, our simulator works without errors
                    although we cannot guarantee that our simulator recognizes the same configs if a users' optimizer
                    slightly changes the content of eval_config.
        """
        trial_info = self.optimizer.ask()
        eval_config = dict(trial_info.config)
        fidels = {self.optimizer.task.fidelity_type: trial_info.budget} if trial_info.budget else None
        config_id = None
        self.history[trial_info.config] = trial_info
        return eval_config, fidels, config_id

    def tell(
        self,
        eval_config: dict[str, Any],
        results: dict[str, float],
        *,
        fidels: dict[str, int | float] | None = None,
        config_id: int | None = None,
    ) -> None:
        """The tell method to register for a tuple of configuration, fidelity, and the results to an optimizer.

        Args:
            eval_config (dict[str, Any]):
                The configuration to be used in the objective function.
            results (dict[str, float]):
                The dict of the return values from the objective function.
            fidels (dict[str, Union[float, int] | None):
                The fidelities to be used in the objective function. Typically training epoch in deep learning.
                If None, we assume that no fidelity is used.
            config_id (int | None):
                The identifier of configuration if needed for continual learning.
                Not used at all when continual_max_fidel=None.
                As we internally use a hash of eval_config, it may be unstable if eval_config has float.
                However, even if config_id is not provided, our simulator works without errors
                although we cannot guarantee that our simulator recognizes the same configs if a users' optimizer
                slightly changes the content of eval_config.

        Returns:
            None
        """
        config = Configuration(values=eval_config, configuration_space=self.optimizer.problem.configspace)
        trial_info_ask = self.history.pop(config)
        trial_info = TrialInfo(
            config=Configuration(values=eval_config, configuration_space=self.optimizer.problem.configspace),
            budget=next(iter(fidels.values())) if fidels else None,
            instance=trial_info_ask.instance,
            seed=trial_info_ask.seed,
            name=trial_info_ask.name,
            checkpoint=trial_info_ask.checkpoint,
        )
        time = None
        if "runtime" in results:
            time = results["runtime"]
            del results["runtime"]
        cost = list(results.values())
        if len(cost) == 1:
            cost = cost[0]

        trial_value = TrialValue(cost=cost, time=time)
        self.optimizer.tell(trial_info=trial_info, trial_value=trial_value)
