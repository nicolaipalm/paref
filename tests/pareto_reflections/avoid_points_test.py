import numpy as np

from paref.pareto_reflections.avoid_points import AvoidPoints


def test_reflecting_example_case():
    nadir, epsilon_avoiding_points, epsilon = np.array([3, 7]), np.array([[2, 1], [1, 5]]), 1
    pareto_reflection = AvoidPoints(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points, epsilon=epsilon)
    assert ((pareto_reflection(np.ones(2)) == nadir).all())


def test_non_dominating_case():
    nadir, epsilon_avoiding_points, epsilon = np.array([3, 7]), np.array([[1, 5], [2, 1]]), 1
    pareto_reflection = AvoidPoints(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points, epsilon=epsilon)

    case = np.ones(2)
    assert ((pareto_reflection(case) == nadir).all())


def test_dominating_case():
    nadir, epsilon_avoiding_points, epsilon = np.array([3, 7]), np.array([5, 1]), 1
    pareto_reflection = AvoidPoints(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points, epsilon=epsilon)

    case = np.ones(2)
    assert ((pareto_reflection(case) == nadir).all())
