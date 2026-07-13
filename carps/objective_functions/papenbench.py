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


import os
import socket

from bencherscaffold.client import BencherClient
from bencherscaffold.protoclasses.bencher_pb2 import Value, ValueType
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer

from carps.utils.env_vars import CARPS_ROOT
from carps.utils.loggingutils import get_logger, setup_logging

setup_logging()
logger = get_logger(__file__)

PAPENBENCH_CONTAINER_FILE = CARPS_ROOT / "build/papenbench.sif"

os.environ["PATH"] = "/opt/software/pc2/EB-SW/software/Apptainer/1.3.5-GCCcore-13.3.0/bin:" + os.environ.get("PATH", "")


def find_free_port() -> int:
    """Helper to find an open high-port on the host node dynamically."""
    # Handle parallel pytest workers
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is not None:
        worker_idx = int(worker_id.replace("gw", ""))
        return 50000 + (worker_idx * 20)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def config_to_value(config: Configuration, hp_type: str) -> list[Value]:
    """Convert ConfigSpace configuration to values.

    Values are the format required by the underlying objective function.

    Args:
        config (Configuration): The configuration to convert.
        hp_type (str): The domain benchmark category type string.

    Returns:
        list[Value]: List of values corresponding to the configuration.
    """
    values = []

    for hp_name in sorted(config.keys()):
        hp_value = config[hp_name]

        if hp_type == "purely_continuous":
            values.append(Value(type=ValueType.CONTINUOUS, value=float(hp_value)))

        elif hp_type == "purely_binary":
            values.append(Value(type=ValueType.BINARY, value=int(hp_value)))

        elif hp_type == "purely_categorical":
            values.append(Value(type=ValueType.CATEGORICAL, value=int(hp_value)))

        elif hp_type == "purely_integer":
            values.append(Value(type=ValueType.INTEGER, value=int(hp_value)))

        elif hp_type == "mixed":
            dim_idx = int(hp_name[1:])  # Parses string index "x0045" -> 45
            if dim_idx < 50:  # noqa: PLR2004
                values.append(Value(type=ValueType.BINARY, value=int(hp_value)))
            else:
                values.append(Value(type=ValueType.CONTINUOUS, value=float(hp_value)))
        else:
            raise ValueError(f"Hyperparameter configuration mapping type '{hp_type}' is unsupported.")

    return values


def get_benchmark_registry() -> Mapping[str, dict[str, Any]]:
    """Get the benchmark registry from the Bencher server.

    Returns:
        The benchmark registry containing benchmark infos like
        port number, dimension and search space type.
    """
    registry_fn = Path(__file__).parent / "../build/lib/bencher/BencherServer/benchmark-registry.json"  # type: ignore[attr-defined,arg-type]
    with open(registry_fn) as f:
        registry = json.load(f)

    # Handle BBOB and PBO
    patched_registry = {}

    for benchmark_name, benchmark_info in registry.items():
        if benchmark_name.startswith("bbob") and benchmark_info["dimensions"] is None:
            for d in [2, 4, 8, 16, 32]:
                new_info = benchmark_info.copy()
                new_info["dimensions"] = d
                patched_registry[f"{benchmark_name}_{d}"] = new_info
        elif benchmark_name.startswith("pbo") and benchmark_info["dimensions"] is None:
            for d in [4, 9, 16, 25, 36]:  # Dim needs to be perfect square
                new_info = benchmark_info.copy()
                new_info["dimensions"] = d
                patched_registry[f"{benchmark_name}_{d}"] = new_info
        else:
            patched_registry[benchmark_name] = benchmark_info

    return patched_registry


def get_configspace(benchmark_name: str) -> ConfigurationSpace:  # noqa: C901, PLR0912
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

    bo4mob_scenarios = ["1ramp", "2corridor", "3junction", "4smallRegion", "5fullRegion"]
    is_bo4mob = any(scen in benchmark_name for scen in bo4mob_scenarios)
    if is_bo4mob:
        hp_type = "purely_integer"

    hps = []

    if hp_type == "purely_continuous":
        hps = [Float(name=f"x{dim:04d}", bounds=(0.0, 1.0)) for dim in range(dimensions)]
        cs.add(hps)

    elif hp_type in ("purely_categorical", "purely_binary"):
        items = [0, 1, 2, 3, 4] if hp_type == "purely_categorical" else [0, 1]
        hps = [Categorical(name=f"x{dim:04d}", items=items) for dim in range(dimensions)]
        cs.add(hps)

    elif hp_type == "mixed":
        if "svmmixed" in benchmark_name.lower():
            # First 50 dimensions are binary
            for dim in range(50):
                hps.append(Categorical(name=f"x{dim:04d}", items=[0, 1]))

            # Last 3 dimensions are continuous
            for dim in range(3):
                hps.append(Float(name=f"x{50 + dim:04d}", bounds=(0.0, 1.0)))

            cs.add(hps)
        else:
            raise NotImplementedError(f"Mixed configuration profile for '{benchmark_name}' is not supported.")

    elif hp_type == "purely_integer":
        # Handle paper-specific integer bounds for BO4Mob configurations
        if is_bo4mob:
            if "1ramp" in benchmark_name:
                lower, upper = 1, 2500
            else:
                lower, upper = 1, 2000
        else:
            raise NotImplementedError(f"Pure integer configuration profile for '{benchmark_name}' is not supported.")

        hps = [Integer(name=f"x{dim:04d}", bounds=(lower, upper)) for dim in range(dimensions)]
        cs.add(hps)

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

        registry = get_benchmark_registry()
        self.benchmark_info = registry.get(benchmark_name)
        if self.benchmark_info is None:
            raise ValueError(f"Benchmark '{benchmark_name}' not found in registry.")

        self.cs = get_configspace(benchmark_name)

        self.instance_id = f"instance_{uuid.uuid4().hex[:8]}"

        # Dynamic port allocation for supporting multiple instances
        self.port = find_free_port()

        self.start_container()
        time.sleep(3)  # Sanity check

        # Create a client to communicate with the Bencher server
        self.client = BencherClient(port=self.port)

    @property
    def configspace(self) -> ConfigurationSpace:
        """Get the configuration space for the benchmark."""
        return get_configspace(self.benchmark_name)

    def start_container(self) -> None:
        """Start the container for the benchmark."""
        start_command = (
            f"apptainer instance start --contain --writable-tmpfs --fakeroot "
            f"{PAPENBENCH_CONTAINER_FILE} {self.instance_id}"
        )

        env = os.environ.copy()

        env["APPTAINERENV_PYTHONUNBUFFERED"] = "1"
        env["APPTAINERENV_OMP_NUM_THREADS"] = "1"
        env["APPTAINERENV_MKL_NUM_THREADS"] = "1"
        env["APPTAINERENV_OPENBLAS_NUM_THREADS"] = "1"
        env["APPTAINERENV_VECLIB_MAXIMUM_THREADS"] = "1"
        env["APPTAINERENV_NUMEXPR_NUM_THREADS"] = "1"

        base_port = self.port
        logger.info(f"Assigning dynamic port ecosystem starting at base: {base_port}")

        # Force-inject the exact environment names defined in the configuration specs
        env["APPTAINERENV_BENCHER_SERVER_PORT"] = str(base_port)
        env["APPTAINERENV_BENCHER_LASSO_PORT"] = str(base_port + 2)
        env["APPTAINERENV_BENCHER_NODEP_PORT"] = str(base_port + 3)
        env["APPTAINERENV_BENCHER_MAXSAT_PORT"] = str(base_port + 4)
        env["APPTAINERENV_BENCHER_EBO_PORT"] = str(base_port + 5)
        env["APPTAINERENV_BENCHER_MUJOCO_PORT"] = str(base_port + 6)
        env["APPTAINERENV_BENCHER_SVM_PORT"] = str(base_port + 7)
        env["APPTAINERENV_BENCHER_IOH_PORT"] = str(base_port + 8)
        env["APPTAINERENV_BENCHER_BO4MOB_PORT"] = str(base_port + 9)

        try:
            subprocess.run(start_command.split(" "), shell=False, check=True, env=env)
            logger.info(f"Start command issued for instance '{self.instance_id}'.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start instance '{self.instance_id}': {e}")
            self.stop_container()
            raise e

        wait_for_instance(self.instance_id)

    def stop_container(self) -> None:
        """Stop the container for the benchmark."""
        stop_command = f"apptainer instance stop {self.instance_id}"
        try:
            result = subprocess.run(
                "apptainer instance list".split(" "),
                shell=False,
                capture_output=True,
                text=True,
                check=False,
            )
            if self.instance_id in result.stdout:
                subprocess.run(stop_command.split(" "), shell=False, check=True)
                logger.info(f"Stop command issued for instance '{self.instance_id}'.")
                wait_for_instance(self.instance_id, wait_for_stop=True)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to stop instance '{self.instance_id}': {e}")

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        # Evaluate the benchmark with the given values
        # This will send the values to the server and return the result
        # If the server is not running, it will raise an error

        values = config_to_value(trial_info.config, hp_type=self.benchmark_info["type"])  # type: ignore[index]

        # Handle dim suffix
        if self.benchmark_name.startswith("bbob") or self.benchmark_name.startswith("pbo"):
            bench = self.benchmark_name.rsplit("_", 1)[0]
        else:
            bench = self.benchmark_name

        starttime = time.time()
        result = self.client.evaluate_point(
            benchmark_name=bench,
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
