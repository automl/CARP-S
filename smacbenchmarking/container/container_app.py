import json
import sys
import os

from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import json as cs_json
from flask import Flask, request
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig

from smacbenchmarking.run import make_problem
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


if (job_id := os.environ['BENCHMARKING_JOB_ID']) != '':
    with open(f"{job_id}_config.txt", 'r') as f:
        hydra_config_path = f.read()

    initialize(version_base=None, config_path="conf")
    cfg = compose(hydra_config_path)

    problem = make_problem(cfg=cfg)

    app = Flask(__name__)


    @app.route("/configspace", methods=["GET"])
    def _request_configspace():
        return json.dumps(cs_json.write(problem.configspace))


    @app.route("/evaluate", methods=["POST"])
    def _request_evaluation():
        if request.is_json:
            trial_info = TrialInfo(**json.loads(request.get_json()))
            return json.dumps(problem.evaluate(trial_info).to_json())
        else:
            raise ValueError("Request is not JSON.")
