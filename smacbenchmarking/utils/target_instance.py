from __future__ import annotations

from typing import Any

from ConfigSpace import Configuration, ConfigurationSpace


class TargetInstance(object):
    def __init__(
        self,
        target_function: callable,
        configuration_space: ConfigurationSpace,
        x_optimum: list[float] | None = None,
        y_optimum: float | None = None,
    ) -> None:
        """Target Instance

        Unified interface to interact with SMAC.

        Parameters
        ----------
        target_function : callable
            The function to optimize.
        configuration_space : ConfigurationSpace
            Configuration space.
        x_optimum : list[float] | None, optional
            Optimal configuration, by default None
        y_optimum : float | None, optional
            Optimal cost, by default None
        """
        self.target_function = target_function
        self.configuration_space = configuration_space
        self.x_optimum = x_optimum
        self.y_optimum = y_optimum

    def __call__(self, configuration: Configuration, seed: int | None = None) -> Any:
        return self.target_function(configuration=configuration, seed=seed)

    def __str__(self) -> str:
        rep = f"Target function: '{self.target_function}', configuration space: {self.configuration_space}"
        return rep
