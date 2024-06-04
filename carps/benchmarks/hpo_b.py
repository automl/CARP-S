"""HPO-B is a benchmark for assessing the performance of black-box HPO algorithms.
Originally from HPO-B Benchmark: https://github.com/releaunifreiburg/HPO-B
Sebastian Pineda-Arango, Hadi S. Jomaa, Martin Wistuba, Josif Grabocka: HPO-B: A Large-Scale Reproducible Benchmark for
Black-Box HPO based on OpenML. NeurIPS Datasets and Benchmarks 2021
Note that this benchmark only converges the surrogate benchmark. Tabular benchmark can not be used in our framework.

To run this benchmark, you must download the benchmark surrogate model under
https://rewind.tf.uni-freiburg.de/index.php/s/rTwPgaxS2Z7NH39/download/saved-surrogates.zip
for details, please refer to https://github.com/releaunifreiburg/HPO-B
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xgboost as xgb
from ConfigSpace import ConfigurationSpace

from carps.benchmarks.problem import Problem
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger

HPOB_SEARCH_SPACE_DIMS = {
    "151": 1,
    "5886": 9,
    "5918": 13,
    "5920": 20,
    "5923": 16,
    "5926": 10,
    "5978": 17,
    "6007": 15,
    "6073": 16,
    "6105": 53,
    "6136": 8,
    "6140": 19,
    "6154": 17,
    "6183": 41,
    "6447": 2,
    "6458": 3,
    "534": 1,
    "4796": 3,
    "5499": 12,
    "5988": 31,
    "5584": 6,
    "189": 1,
    "7064": 8,
    "5527": 8,
    "5636": 6,
    "5859": 6,
    "5860": 2,
    "5890": 2,
    "5891": 8,
    "5906": 16,
    "5965": 10,
    "5968": 8,
    "5970": 2,
    "5971": 16,
    "6308": 2,
    "6762": 6,
    "6766": 2,
    "6767": 18,
    "6794": 10,
    "6856": 1,
    "7188": 1,
    "7189": 3,
    "7190": 3,
    "7607": 9,
    "7609": 9,
    "5624": 7,
    "124": 1,
    "153": 1,
    "243": 1,
    "245": 1,
    "246": 1,
    "247": 1,
    "248": 1,
    "423": 1,
    "506": 1,
    "633": 1,
    "2553": 1,
    "4006": 1,
    "5526": 2,
    "7286": 2,
    "7290": 4,
    "5626": 17,
    "5889": 6,
    "5458": 16,
    "5489": 14,
    "4828": 10,
    "214": 1,
    "6741": 21,
    "158": 1,
    "2566": 2,
    "3894": 6,
    "6003": 18,
    "6765": 2,
    "5963": 15,
    "6000": 2,
    "6322": 8,
    "5623": 3,
    "5969": 8,
    "3737": 2,
    "829": 4,
    "833": 5,
    "935": 4,
    "6323": 2,
    "7200": 19,
    "673": 1,
    "674": 1,
    "678": 1,
    "679": 1,
    "680": 1,
    "681": 1,
    "682": 1,
    "683": 1,
    "684": 1,
    "685": 1,
    "688": 1,
    "689": 1,
    "690": 1,
    "691": 1,
    "692": 1,
    "693": 1,
    "694": 1,
    "695": 1,
    "696": 1,
    "697": 1,
    "3994": 1,
    "5964": 9,
    "5972": 2,
    "6075": 6,
    "2010": 8,
    "2039": 14,
    "2073": 13,
    "2277": 15,
    "3489": 12,
    "3490": 29,
    "3502": 6,
    "3960": 15,
    "4289": 20,
    "5218": 19,
    "5237": 8,
    "5253": 2,
    "5295": 13,
    "5301": 15,
    "5315": 9,
    "2614": 2,
    "2629": 6,
    "2793": 12,
    "2799": 7,
    "2823": 11,
    "3414": 2,
    "3425": 9,
    "3434": 19,
    "3439": 3,
    "3442": 5,
    "5503": 2,
    "6131": 2,
    "5435": 3,
    "7021": 3,
    "5502": 8,
    "5521": 3,
    "5604": 12,
    "5704": 3,
    "5788": 3,
    "5813": 10,
    "7680": 2,
    "7604": 2,
    "5919": 5,
    "5921": 3,
    "5922": 10,
    "5960": 7,
    "6024": 10,
    "6124": 18,
    "6134": 11,
    "6137": 18,
    "6139": 16,
    "6155": 7,
    "6156": 1,
    "6182": 3,
    "6189": 20,
    "6190": 6,
    "6211": 15,
    "6212": 4,
    "6213": 3,
    "6215": 10,
    "6216": 5,
    "6271": 18,
    "6285": 10,
    "6309": 7,
    "6345": 20,
    "6347": 2,
    "6364": 3,
    "6365": 4,
    "6376": 7,
    "6433": 18,
    "6461": 8,
    "6493": 15,
    "6507": 3,
}


class HPOBProblem(Problem):
    def __init__(
        self,
        dataset_id: tuple[str, int],
        model_id: tuple[str, int],
        surrogates_dir: Path = Path("carps/benchmark_data/HPO-B/saved-surrogates"),
        loggers: list[AbstractLogger] | None = None,
    ):
        """Constructor for the HPO-B handler. Given that the configuration space of HPO-B tabular dataset is not generated
        from grid, we only consider surrogate benchmark.
        Parameters.
        ----------
            dataset_id: tuple[str, int]
                dataset id, the ids can be found under surrogate_model summary directory
            model_id: tuple[str, int]
                model id, each model corresponds to a search space, the search space dimensions can be found under
            surrogates_dir: Path
                path to directory with surrogates models.
        """
        super().__init__(loggers)
        self.model_id = str(model_id)
        self.dataset_id = str(dataset_id)

        surrogates_dir = Path(surrogates_dir)
        surrogates_file = surrogates_dir / "summary-stats.json"
        if not surrogates_file.is_file():
            raise RuntimeError(
                "It seems that the surrogate files have not been downloaded. Please run "
                "'bash container_recipes/benchmarks/hpob/download_data.sh' to download the "
                "surrogates."
            )
        with open(str(surrogates_file)) as f:
            self.surrogates_stats = json.load(f)
        self.surrogate_dir = surrogates_dir

        self.surrogate_model = self._get_surrogate_model(self.dataset_id, self.model_id)

        # generate configuration space, all the feature values range from [0,1] according to the setting
        search_space_dims = HPOB_SEARCH_SPACE_DIMS[model_id]
        self.search_space_dims = search_space_dims

        self._configspace = self._get_configspace(search_space_dims)

    def _get_surrogate_model(self, dataset_id: str, model_id: str) -> xgb.Booster:
        """Get the surrogate model for the problem."""
        surrogate_name = "surrogate-" + str(model_id) + "-" + str(dataset_id)
        surrogate_dir = self.surrogate_dir / (surrogate_name + ".json")
        if not surrogate_dir.exists():
            raise ValueError(f"Unknown dataset: {dataset_id} and model: {model_id} combination")
        bst_surrogate = xgb.Booster()
        bst_surrogate.load_model(str(self.surrogate_dir / (surrogate_name + ".json")))
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

        Returns:
        -------
        ConfigurationSpace
            Configuration space.
        """
        return self._configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        """Evaluate problem.

        Parameters
        ----------
        trial_info : TrialInfo
            Dataclass with configuration, seed, budget, instance.

        Returns:
        -------
        TrialValue
            Cost
        """
        configuration = trial_info.config
        input = np.asarray(list(dict(configuration).values()))
        starttime = time.time()
        x_q = xgb.DMatrix(input.reshape(-1, self.search_space_dims))
        predicted_output = self.surrogate_model.predict(x_q)
        endtime = time.time()
        T = endtime - starttime

        # we would like to do minimization
        return TrialValue(cost=-predicted_output.item(), time=T, starttime=starttime, endtime=endtime)
