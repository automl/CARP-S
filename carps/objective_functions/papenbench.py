"""Objective function for PapenBench benchmarks.

Source github: https://github.com/LeoIV/bencher/
Paper: https://arxiv.org/abs/2505.21321
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from carps.objective_functions.objective_function import ObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from carps.loggers.abstract_logger import AbstractLogger

import importlib.util

from bencherscaffold.client import BencherClient
from bencherscaffold.protoclasses.bencher_pb2 import Value, ValueType
from ConfigSpace import Configuration, ConfigurationSpace, Float, UniformFloatHyperparameter

from carps.utils.env_vars import CARPS_ROOT
from carps.utils.loggingutils import get_logger, setup_logging

setup_logging()
logger = get_logger(__file__)


PAPENBENCH_CONTAINER_FILE = CARPS_ROOT / "build/papenbench.sif"


def config_to_value(config: Configuration, configspace: ConfigurationSpace) -> list[Value]:
    """Convert ConfigSpace configuration to values.

    Values are the format required by the underlying objective function.

    Args:
        config (Configuration): The configuration to convert.
        configspace (ConfigurationSpace): The associated configuration space.
            Necessary to infer the hyperparameter type.

    Returns:
        list[Value]: List of values corresponding to the configuration.
    """
    values = []

    for hp, hp_value in config.items():
        if isinstance(configspace[hp], UniformFloatHyperparameter):
            values.append(Value(type=ValueType.CONTINUOUS, value=hp_value))
        else:
            raise ValueError(f"Hp type not supported: {type(hp)}")
    return values


def get_benchmark_registry() -> Mapping[str, dict[str, Any]]:
    """Get the benchmark registry from the Bencher server.

    Returns:
        The benchmark registry containing benchmark infos like
        port number, dimension and search space type.
    """
    spec = importlib.util.find_spec("bencherserver")
    assert spec is not None, "bencherserver package not found. Please install it."
    registry_fn = Path(spec.origin).parent / "benchmark-registry.json"  # type: ignore[attr-defined,arg-type]
    with open(registry_fn) as f:
        return json.load(f)


def get_configspace(benchmark_name: str) -> ConfigurationSpace:
    """Get configuration space for specific objective function.

    Args:
        benchmark_name (str): Name of the benchmark.

    Returns:
        ConfigurationSpace
            The search space.
    """
    registry = get_benchmark_registry()
    benchmark_info = registry.get(benchmark_name)
    if benchmark_info is None:
        raise ValueError(f"Benchmark '{benchmark_name}' not found in registry.")

    hp_type = benchmark_info["type"]
    dimensions = benchmark_info["dimensions"]
    cs = ConfigurationSpace()
    if hp_type == "purely_continuous":
        for dim in range(dimensions):
            hp_float = Float(
                name=f"x{dim:04d}",
                bounds=(0.0, 1.0),
            )
            cs.add(hp_float)
    else:
        raise NotImplementedError(f"Benchmark type '{hp_type}' is not supported.")
    return cs


def wait_for_instance(instance_id: str, timeout: float = 30, interval: float = 1, wait_for_stop: bool = False) -> bool:  # noqa: FBT001, FBT002
    """Wait for apptainer instance to start or stop.

    Args:
        instance_id (str): The ID of the instance to wait for.
        timeout (float): Maximum time to wait in seconds.
        interval (float): Time to wait between checks in seconds.
        wait_for_stop (bool): If True, waits for the instance to stop instead of starting.

    Returns:
        bool: True if the instance started or stopped within the timeout, False otherwise.
    """
    logger.debug(f"Waiting up to {timeout} seconds for instance '{instance_id}' to start...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        result = subprocess.run(
            "apptainer instance list".split(" "),
            shell=False,
            capture_output=True,
            text=True,
            check=False,
        )
        logger.debug(f"Instance list output:\n{result.stdout}")
        if instance_id in result.stdout and not wait_for_stop:
            logger.info(f"Instance '{instance_id}' is now running.")
            return True
        if wait_for_stop and instance_id not in result.stdout:
            logger.info(f"Instance '{instance_id}' has stopped.")
            return True
        time.sleep(interval)
    logger.warning(f"Timeout: Instance '{instance_id}' did not appear in time.")
    return False


class PapenBenchObjectiveFunction(ObjectiveFunction):
    """PapenBench ObjectiveFunction class."""

    def __init__(
        self,
        benchmark_name: str,
        loggers: list[AbstractLogger] | None = None,
    ):
        """Initialize the PapenBench objective function.

        Args:
            benchmark_name (str): Name of the benchmark to run.
            loggers (list[AbstractLogger] | None): List of loggers to use for logging.
        """
        super().__init__(loggers=loggers)

        self.benchmark_name = benchmark_name

        self.cs = get_configspace(benchmark_name)

        self.instance_id = f"instance_{uuid.uuid4().hex[:8]}"

        self.start_container()

        # Create a client to communicate with the Bencher server
        # By default, it connects to 127.0.0.1:50051
        self.client = BencherClient()

    @property
    def configspace(self) -> ConfigurationSpace:
        """Get the configuration space for the benchmark."""
        return get_configspace(self.benchmark_name)

    def start_container(self) -> None:
        """Start the container for the benchmark."""
        start_command = f"apptainer instance start {PAPENBENCH_CONTAINER_FILE} {self.instance_id}"
        try:
            subprocess.run(start_command.split(" "), shell=False, check=True)
            logger.info(f"Start command issued for instance '{self.instance_id}'.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start instance '{self.instance_id}': {e}")
            raise e

        wait_for_instance(self.instance_id)

    def stop_container(self) -> None:
        """Stop the container for the benchmark."""
        stop_command = f"apptainer instance stop {self.instance_id}"

        result = subprocess.run(
            "apptainer instance list".split(" "),
            shell=False,
            capture_output=True,
            text=True,
            check=False,
        )
        logger.debug(f"Instance list output:\n{result.stdout}")
        if self.instance_id in result.stdout:
            try:
                subprocess.run(stop_command.split(" "), shell=False, check=True)
                logger.info(f"Stop command issued for instance '{self.instance_id}'.")

                wait_for_instance(self.instance_id, wait_for_stop=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to stop instance '{self.instance_id}': {e}")
                raise e
        else:
            logger.warning(f"Instance '{self.instance_id}' is not running, no stop command issued.")

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        # Evaluate the benchmark with the given values
        # This will send the values to the server and return the result
        # If the server is not running, it will raise an error

        values = config_to_value(trial_info.config, configspace=self.cs)

        starttime = time.time()
        result = self.client.evaluate_point(
            benchmark_name=self.benchmark_name,
            point=values,
        )
        endtime = time.time()
        time_elapsed = endtime - starttime
        logger.info(f"Evaluation result: {result}")
        return TrialValue(
            cost=result,
            time=time_elapsed,
            starttime=starttime,
            endtime=endtime,
        )

    def __del__(self) -> None:
        """Ensure the container is stopped when the object is deleted."""
        self.stop_container()
