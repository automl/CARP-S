"""Utils for handling ConfigSpace configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ConfigSpace.hyperparameters.hyperparameter import FloatHyperparameter

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace


def clip_bounds(configuration: dict, config_space: ConfigurationSpace) -> dict:
    """Clip bounds of configuration space.

    This is necessary because some optimizers ask to evaluate configurations which are numerically **just** out of
    bounds, which can happen after log transformations.

    Parameters
    ----------
    configuration : dict
        Configuration to clip.
    config_space : ConfigSpace
        Associated configuration space.

    Returns:
    -------
    dict
        Clipped configuration.
    """
    for name in configuration:
        hp = config_space[name]
        if isinstance(hp, FloatHyperparameter):
            value = configuration[name]
            configuration[name] = max(min(value, hp.upper), hp.lower)
    return configuration
