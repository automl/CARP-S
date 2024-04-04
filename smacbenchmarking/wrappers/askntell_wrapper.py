from __future__ import annotations

import time

from smacbenchmarking.wrappers.optimizer_wrapper import OptimizerWrapper


class AskAndTellWrapper(OptimizerWrapper):
    def run(self) -> None:
        """Run Ask and Tell Optimization
        """
        if self.solver is None:
            self.setup_optimizer()

        self.trial_counter: int = 0

        self.start_time = time.time()
        while True:
            if self.n_trials is not None:
                if self.trial_counter >= self.n_trials:
                    break
            if self.time_budget is not None:
                if time.time() - self.start_time > self.time_budget:
                    break

            trial_info = self.ask()
            trial_value = self.optimizer.problem.evaluate(trial_info=trial_info)
            self.tell(trial_value=trial_value)

            self.trial_counter += 1

        return None
