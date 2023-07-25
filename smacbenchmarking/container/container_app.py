import json
import sys

from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import json as cs_json
from flask import Flask, request

from smacbenchmarking.utils.trials import TrialInfo, TrialValue

# command line arg
config_path = sys.argv[1]

configspace = ConfigurationSpace()

app = Flask(__name__)


# TODO: remove (dummy app)
def evaluate(trial_info: TrialInfo):
    return TrialValue(cost=42)


@app.route("/configspace", methods=["GET"])
def _request_configspace():
    return json.dumps(cs_json.write(configspace))


@app.route("/evaluate", methods=["POST"])
def _request_evaluation():
    if request.is_json:
        trial_info = TrialInfo(**json.loads(request.get_json()))
        return json.dumps(evaluate(trial_info).to_json())
    else:
        raise ValueError("Request is not JSON.")
