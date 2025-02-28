# Contributing a Benchmark

To add a new benchmark problem to CARP-S, you need to create a Python file that defines a 
new problem class. This class should inherit from the `ObjectiveFunction` class defined in 
`carps/benchmarks/problem.py`. 

Here's a step-by-step guide for how to add a new benchmark:

1. **Benchmark Python file**:
Create a new Python file in the `carps/benchmarks/` directory. 
For example, you might name it `my_benchmark.py`.


2. **Define your problem class**: 
Define a new class that inherits from `ObjectiveFunction`. This class should implement the `configspace` 
property and the `_evaluate` method, as these are abstract in the base `ObjectiveFunction` class. 
The `configspace` property should return a `ConfigurationSpace` object that defines the 
configuration space for your problem. The `_evaluate` method should take a `TrialInfo` object 
and return a `TrialValue` object. If your problem requires additional methods, you can implement 
them in your class. For example, you might need a method to load data for your problem. 


3. **Requirements file**: Create a requirements file and add the requirements for your benchmark. 
   The file structure must be 
   `container_recipes/benchmarks/<benchmark_id>/<benchmark_id>_requirements.txt`, so for example,
   `container_recipes/benchmarks/my_benchmark/my_benchmark_requirements.txt`. Please specify exact 
   versions of all requirements! This is very important for reproducibility.


4. **Config files**: Add config files for the different benchmarking tasks under 
   `carps/configs/problem/my_benchmark/my_benchmark_config_{task}.yaml`. 
   You can use the existing config files as a template.

Here's a basic example of what your `my_benchmark.py` file might look like:

```python
from ConfigSpace import ConfigurationSpace
from carps.benchmarks.problem import ObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue

class MyBenchmarkObjectiveFunction(ObjectiveFunction):
    def __init__(self, loggers=None):
        super().__init__(loggers)
        # Initialize any additional attributes your problem needs here

    @property
    def configspace(self) -> ConfigurationSpace:
        # Return a ConfigurationSpace object that defines the configuration space for your problem
        pass

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        # Evaluate a trial and return a TrialValue object
        pass

    # Implement any additional methods your problem needs here
```
