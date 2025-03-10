"""Container script for the task container."""

from __future__ import annotations

import json
import os

from ConfigSpace.read_and_write import json as cs_json
from flask import Flask, request  # type: ignore
from omegaconf import OmegaConf

from carps.utils.loggingutils import CustomEncoder
from carps.utils.running import make_task
from carps.utils.trials import TrialInfo

if (job_id := os.environ["BENCHMARKING_JOB_ID"]) != "":
    cfg = OmegaConf.load(f"{job_id}_hydra_config.yaml")

task = make_task(cfg=cfg)

# TODO Check that task container and task match

app = Flask(__name__)
app.run()


@app.route("/configspace", methods=["GET"])  # type: ignore[misc]
def _request_configspace() -> str:
    return json.dumps(cs_json.write(task.objective_function.configspace), cls=CustomEncoder)


@app.route("/evaluate", methods=["POST"])  # type: ignore[misc]
def _request_evaluation() -> str:
    if request.is_json:
        trial_info = TrialInfo(**json.loads(request.get_json()))
        trial_value = task.objective_function.evaluate(trial_info)
        return json.dumps(trial_value.to_json(), cls=CustomEncoder)  # type: ignore[attr-defined]
    raise ValueError("Request is not JSON.")
