# Contributing a Benchmark

To add a new benchmark problem to CARP-S, you need to create a Python file that defines a 
new problem class. This class should inherit from the `Problem` class defined in 
`carps/benchmarks/problem.py`. 

Here's a step-by-step guide for how to add a new benchmark:

1. **Create a new Python file**:
Create a new Python file in the `carps/benchmarks/` directory. 
For example, you might name it `my_benchmark.py`.

2. **Define your problem class**: 
Define a new class that inherits from `Problem`. This class should implement the `configspace` 
property and the `_evaluate` method, as these are abstract in the base `Problem` class. 
The `configspace` property should return a `ConfigurationSpace` object that defines the 
configuration space for your problem. The `_evaluate` method should take a `TrialInfo` object 
and return a `TrialValue` object.

3. **Implement additional methods**: If your problem requires additional methods, you can implement 
them in your class. For example, you might need a method to load data for your problem.

Here's a basic example of what your `my_benchmark.py` file might look like:

```python
from ConfigSpace import ConfigurationSpace
from carps.benchmarks.problem import Problem
from carps.utils.trials import TrialInfo, TrialValue

class MyBenchmarkProblem(Problem):
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
