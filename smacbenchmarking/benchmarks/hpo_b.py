"""
HPO-B is a benchmark for assessing the performance of black-box HPO algorithms.
Originally from HPO-B Benchmark: https://github.com/releaunifreiburg/HPO-B
Sebastian Pineda-Arango, Hadi S. Jomaa, Martin Wistuba, Josif Grabocka: HPO-B: A Large-Scale Reproducible Benchmark for
Black-Box HPO based on OpenML. NeurIPS Datasets and Benchmarks 2021
Note that this benchmark only converges the surrogate benchmark. Tabular benchmark can not be used in our framework

To run this benchmark, you must download the benchmark surrogate model under
https://rewind.tf.uni-freiburg.de/index.php/s/rTwPgaxS2Z7NH39/download/saved-surrogates.zip
for details, please refer to https://github.com/releaunifreiburg/HPO-B
"""

from __future__ import annotations

from pathlib import Path
import json

import xgboost as xgb

from ConfigSpace import ConfigurationSpace
from smac.runhistory.dataclasses import TrialInfo

from smacbenchmarking.benchmarks.problem import SingleObjectiveProblem

HPOB_SEARCH_SPACE_DIMS = {"4796": 3,
                          "5527": 8,
                          "5636": 6,
                          "5859": 6,
                          "5860": 2,
                          "5891": 8,
                          "5906": 16,
                          "5965": 10,
                          "5970": 2,
                          "5971": 16,
                          "6766": 2,
                          "6767": 18,
                          "6794": 10,
                          "7607": 9,
                          "7609": 9,
                          "5889": 6}


class HPOBProblem(SingleObjectiveProblem):
    def __init__(self,
                 dataset_id: tuple[str, int],
                 model_id: tuple[str, int],
                 surrogates_dir: Path = Path("saved-surrogates")):
        """
        Constructor for the HPO-B handler. Given that the configuration space of HPO-B tabular dataset is not generated
        from grid, we only consider surrogate benchmark.
        Parameters
        ----------
            dataset_id: tuple[str, int]
                dataset id, the ids can be found under surrogate_model summary directory
            model_id: tuple[str, int]
                model id, each model corresponds to a search space, the search space dimensions can be found under
            surrogates_dir: Path
                path to directory with surrogates models.
        """
        super().__init__()
        self.model_id = str(model_id)
        self.dataset_id = str(dataset_id)

        surrogates_file = surrogates_dir / "summary-stats.json"
        with open(str(surrogates_file)) as f:
            self.surrogates_stats = json.load(f)
        self.surrogate_dir = surrogates_dir

        self.surrogate_model = self._get_surrogate_model(self.dataset_id, self.model_id)

        # generate configuration space, all the feature values range from [0,1] according to the setting
        search_space_dims = HPOB_SEARCH_SPACE_DIMS[model_id]
        self.search_space_dims = search_space_dims

        self._configspace = self._get_configspace(search_space_dims)

    def _get_surrogate_model(self, dataset_id: str, model_id: str) -> xgb.Booster:
        """
        Get the surrogate model for the problem
        """
        surrogate_name = 'surrogate-' + str(model_id) + '-' + str(dataset_id)
        surrogate_dir = self.surrogate_dir / (surrogate_name + '.json')
        if not surrogate_dir.exists():
            raise ValueError(f"Unknown dataset: {dataset_id} and model: {model_id} combination")
        bst_surrogate = xgb.Booster()
        bst_surrogate.load_model(str(self.surrogate_dir / (surrogate_name + '.json')))
        return bst_surrogate

    def _get_configspace(self, search_space_dims: int):
        # generate configuration space, all the feature values range 0 to 1
        bounds = tuple([(0, 1) for _ in range(search_space_dims)])
        cs = ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(bounds)
        return cs

    @property
    def configspace(self) -> ConfigurationSpace:
        """Return configuration space.

        Returns
        -------
        ConfigurationSpace
            Configuration space.
        """
        return self._configspace

    def evaluate(self, trial_info: TrialInfo) -> float:
        """Evaluate problem.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns
        -------
        float
            Cost
        """
        configuration = trial_info.config
        input = list(dict(configuration).values())
        x_q = xgb.DMatrix(input.reshape(-1, self.search_space_dims))
        predicted_output = self.surrogate_model.predict(x_q)
        # we would like to do minimization
        return - predicted_output
