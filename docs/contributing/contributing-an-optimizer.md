# Contributing an Optimizer

To add a new optimizer to CARPS, you need to create a new Python file that defines a new optimizer 
class. This class should inherit from the `Optimizer` class defined in 
`carps/optimizers/optimizer.py`. You can have a look at the 
[optimizer template](https://github.com/automl/CARP-S-template/blob/main/my-optimizer.py) 
for inspiration.

Here's a step-by-step guide for how to add a new optimizer:

1. **Create a new Python file**: 
Create a new Python file in the `carps/optimizers/` directory.
For example, you might name it `my_optimizer.py`.


2. **Define your optimizer class**:
Define a new class that inherits from `Optimizer`. This class should implement the 
`convert_configspace` method, that takes a ConfigSpace configuration space and converts it to 
a search space from the optimizer, and a `convert_to_trial` method, that converts a proposal by 
the optimizer to a TrialInfo object. Furthermore, a `_setup_optimizer` method should be implemented,
setting up and returning the optimizer to be used, and a `get_current_incumbent` method, extracting 
the incumbent config and cost. Finally, an `ask` method is required, that 
queries a new trial to evaluate from the optimizer and returns it as a TrialInfo object, and a 
`tell` method, that takes a TrialInfo and TrialValue and updates the optimizer with the results of 
the trial. If your optimizer requires additional methods, you can implement them in your class. 


3. **Requirements file**: Create a requirements file and add the requirements for your optimizer.
   The file structure must be 
   `container_recipes/optimizers/<optimizer_container_id>/<optimizer_container_id>_requirements.txt`,
   so for example, `container_recipes/optimizers/my_optimizer/my_optimizer_requirements.txt`.
   Please specify exact versions of all requirements! This is very important for reproducibility.


4. **Config files**: Add config files for the different optimizers under 
   `carps/configs/optimizer/my_optimizer/my_optimizer_config_{variant}.yaml`. 
   You can use the existing config files as a template.

Here's a basic example of what your `my_optimizer.py` file might look like:

```python
from ConfigSpace import Configuration, ConfigurationSpace
from carps.utils.trials import TrialInfo, TrialValue
from carps.optimizers.optimizer import Optimizer
from carps.utils.types import Incumbent

class MyOptimizer(Optimizer):
    def __init__(self, problem, task, loggers=None):
        super().__init__(problem, task, loggers)
        # Initialize any additional attributes your optimizer needs here

    def convert_configspace(self, configspace: ConfigurationSpace) -> OptimizerSearchSpace:
        # Convert ConfigSpace configuration space to search space from optimizer.
        pass

    def convert_to_trial(self, optimizer_trial: OptimizerTrial) -> TrialInfo:
        # Convert proposal by optimizer to TrialInfo.
        pass

    def _setup_optimizer(self) -> Any:
        # Setup the optimizer.
        pass

    def get_current_incumbent(self) -> Incumbent:
        # Extract the incumbent config and cost.
        pass

    def ask(self) -> TrialInfo:
        # Ask the optimizer for a new trial to evaluate.
        pass

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        # Tell the optimizer a new trial.
        pass

    # Implement any additional methods your optimizer needs here
```
