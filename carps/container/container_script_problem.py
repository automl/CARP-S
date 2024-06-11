from __future__ import annotations

import json
import os

from ConfigSpace.read_and_write import json as cs_json
from flask import Flask, request
from omegaconf import OmegaConf

from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from carps.utils.loggingutils import CustomEncoder

if (job_id := os.environ["BENCHMARKING_JOB_ID"]) != "":
    cfg = OmegaConf.load(f"{job_id}_hydra_config.yaml")

problem = make_problem(cfg=cfg)

# TODO Check that problem container and problem match

app = Flask(__name__)
app.run()


@app.route("/configspace", methods=["GET"])
def _request_configspace() -> str:
    return json.dumps(cs_json.write(problem.configspace), cls=CustomEncoder)


@app.route("/evaluate", methods=["POST"])
def _request_evaluation() -> str:
    if request.is_json:
        trial_info = TrialInfo(**json.loads(request.get_json()))
        return json.dumps(problem.evaluate(trial_info).to_json(), cls=CustomEncoder)
    else:
        raise ValueError("Request is not JSON.")
