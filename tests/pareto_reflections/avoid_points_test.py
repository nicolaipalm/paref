import numpy as np
import pytest

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


def test_raise_init_value_error_with_input_dimension_mismatch():
    nadir, epsilon_avoiding_points, epsilon = np.array([3, 7]), np.array([5, 2, 1]), 1
    with pytest.raises(ValueError,
                       match=r'.*Nadir and avoiding points need to be 2-dimensional numpy arrays of equal shape!.*'):
        _ = AvoidPoints(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points, epsilon=epsilon)

    nadir, epsilon_avoiding_points, epsilon = np.array([3, 7]), np.array([5, 2]), np.array([5, 2, 1])
    with pytest.raises(ValueError,
                       match=r'.*Epsilon must be a Real Number or a numpy array of same shape as nadir!.*'):
        _ = AvoidPoints(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points, epsilon=epsilon)

    nadir, epsilon_avoiding_points, epsilon = np.array([3, 7]), np.array([5, 2]), -1
    with pytest.raises(ValueError, match=r'.*Epsilon must be positive!.*'):
        _ = AvoidPoints(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points, epsilon=epsilon)


def test_raise_value_error_with_differing_input_shapes():
    nadir, epsilon_avoiding_points, epsilon = np.array([3, 7]), np.array([5, 2]), 1
    with pytest.raises(ValueError,
                       match=r'.*Input x must be of dimension 1! Shape of x is *'):
        pareto_reflection = AvoidPoints(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points, epsilon=epsilon)
        case = np.array([[3, 7], [3, 7]])
        pareto_reflection(case)
