"""Tests for the constrained synthetic objective functions."""

from __future__ import annotations

import numpy as np
import pytest
from carps.objective_functions.constrained_synthetic import (
    PROBLEMS,
    ConstrainedSyntheticObjectiveFunction,
)
from carps.utils.trials import TrialInfo

# The best feasible value of each problem, as published in the paper it comes from. These pin the sign
# and bound conventions, which are easy to state backwards: a flipped inequality still produces a
# plausible-looking landscape, just not the published one.
PUBLISHED_OPTIMA = {
    "branin_disk": 0.397887,  # Gelbart et al.: the disk keeps only the minimum at (pi, 2.275)
    "gardner1": -1.88875,  # Gardner et al., stated there as a maximization of the negated objective
    "gardner2": 0.254,  # Gardner et al.
    "gramacy": 0.5998,  # Gramacy et al.
}


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_the_constrained_optimum_matches_the_published_value(name: str) -> None:
    assert PROBLEMS[name].constrained_optimum(resolution=3000) == pytest.approx(PUBLISHED_OPTIMA[name], abs=1e-3)


@pytest.mark.parametrize("name", ["gardner1", "gardner2", "gramacy"])
def test_the_constraint_costs_something(name: str) -> None:
    """A constraint that does not change the answer would make the problem a poor test of feasibility.

    branin_disk is excluded deliberately: its disk keeps the best value and removes two of the three
    places it is attained, which `test_branin_keeps_one_of_its_three_minima` checks instead.
    """
    problem = PROBLEMS[name]
    axes = [np.linspace(lower, upper, 1000) for lower, upper in problem.bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([m.ravel() for m in mesh], axis=1)

    unconstrained = float(np.min(problem.objective(points.T)))

    assert problem.constrained_optimum(resolution=1000) > unconstrained + 1e-6


def test_the_optimum_location_is_where_branins_surviving_minimum_is() -> None:
    argmin = PROBLEMS["branin_disk"].constrained_argmin(resolution=3000)

    assert argmin["x0"] == pytest.approx(np.pi, abs=1e-2)
    assert argmin["x1"] == pytest.approx(2.275, abs=1e-2)


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_the_optimum_location_is_feasible_and_attains_the_optimum(name: str) -> None:
    problem = PROBLEMS[name]
    argmin = problem.constrained_argmin(resolution=2000)
    point = np.array([argmin[f"x{i}"] for i in range(problem.dimensions)])

    assert problem.is_feasible(point)
    assert float(problem.objective(point)) == pytest.approx(problem.constrained_optimum(2000), abs=1e-6)


def test_branin_keeps_one_of_its_three_minima() -> None:
    """The point of the disk: the best value is unchanged, but two ways of reaching it are removed."""
    problem = PROBLEMS["branin_disk"]
    minima = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]
    feasible = [problem.is_feasible(np.array(m)) for m in minima]

    assert feasible == [False, True, False]


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_the_constrained_quantities_are_reported_alongside_the_cost(name: str) -> None:
    objective_function = ConstrainedSyntheticObjectiveFunction(problem_name=name, seed=0)
    config = objective_function.configspace.sample_configuration()

    trial_value = objective_function.evaluate(TrialInfo(config=config))

    assert isinstance(trial_value.cost, float)
    assert set(trial_value.additional_info) == set(PROBLEMS[name].constraints)
    assert all(isinstance(v, float) for v in trial_value.additional_info.values())


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_the_constraint_expressions_name_the_reported_quantities(name: str) -> None:
    """A bound has to refer to something the objective function actually reports."""
    problem = PROBLEMS[name]
    reported = set(problem.constraints)

    for expression in problem.constraint_expressions:
        assert expression.split()[0] in reported


def test_an_unknown_problem_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown problem"):
        ConstrainedSyntheticObjectiveFunction(problem_name="nope", seed=0)
