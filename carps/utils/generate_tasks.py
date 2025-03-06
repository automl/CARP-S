"""Utils for Task Generation."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from carps.utils.task import InputSpace, OptimizationResources, OutputSpace, TaskMetadata


def get_target_name(obj: Any) -> str:
    """Get target name of an object.

    For hydra object instantiation.

    Args:
        obj (Any): Object.

    Returns:
        str: Target name, e.g. carps.optimizers.random_search.RandomSearchOptimizer.
    """
    return type(obj).__module__ + "." + type(obj).__name__


def get_dict_input_space(input_space: InputSpace) -> dict[str, Any]:
    """Get dictionary representation of input space.

    For hydra yaml serialization.

    Args:
        input_space (InputSpace): Input space.

    Returns:
        dict[str, Any]: Dictionary representation of input space with targets.
    """
    D = {
        "_target_": get_target_name(input_space),
        "configuration_space": {
            "_target_": "ConfigSpace.configuration_space.ConfigurationSpace.from_serialized_dict",
            "d": input_space.configuration_space.to_serialized_dict(),
        },
        "fidelity_space": {
            "_target_": get_target_name(input_space.fidelity_space),
            **asdict(input_space.fidelity_space),
        },
        "instance_space": None,
    }
    if input_space.instance_space is not None:
        D["instance_space"] = {
            "_target_": get_target_name(input_space.instance_space),
            **asdict(input_space.instance_space),
        }
    return D


def get_dict_output_space(output_space: OutputSpace) -> dict[str, Any]:
    """Get dictionary representation of output space.

    For hydra yaml serialization.

    Args:
        output_space (OutputSpace): Output space.

    Returns:
        dict[str, Any]: Dictionary representation of output space with targets.
    """
    return {"_target_": get_target_name(output_space), **asdict(output_space)}


def get_dict_opt_resources(opt_resources: OptimizationResources) -> dict[str, Any]:
    """Get dictionary representation of optimization resources.

    For hydra yaml serialization.

    Args:
        opt_resources (OptimizationResources): Optimization resources.

    Returns:
        dict[str, Any]: Dictionary representation of optimization resources with targets.
    """
    return {"_target_": get_target_name(opt_resources), **asdict(opt_resources)}


def get_dict_metadata(task_metadata: TaskMetadata) -> dict[str, Any]:
    """Get dictionary representation of task metadata.

    For hydra yaml serialization.

    Args:
        task_metadata (TaskMetadata): Task metadata.

    Returns:
        dict[str, Any]: Dictionary representation of task metadata with targets.
    """
    return {"_target_": get_target_name(task_metadata), **asdict(task_metadata)}
