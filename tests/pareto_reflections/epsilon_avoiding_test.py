import numpy as np
import pytest

from paref.pareto_reflections.epsilon_avoiding import EpsilonAvoiding


def test_reflecting_example_case():
    nadir, epsilon_avoiding_points, epsilon = np.array([3, 7]), np.array([[2, 1], [1, 5], [1, 5]]), 1
    pareto_reflection = EpsilonAvoiding(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points, epsilon=epsilon)
    assert ((pareto_reflection(np.ones(2)) == nadir).all())


def test_non_dominating_case():
    nadir, epsilon_avoiding_points, epsilon = np.array([3, 7]), np.array([[1, 5], [2, 1]]), 1
    pareto_reflection = EpsilonAvoiding(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points, epsilon=epsilon)

    case = np.ones(2)
    assert ((pareto_reflection(case) == nadir).all())


def test_dominating_case():
    nadir, epsilon_avoiding_points, epsilon = np.array([3, 7]), np.array([[5, 2], [2, 5]]), 1
    pareto_reflection = EpsilonAvoiding(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points, epsilon=epsilon)

    case = np.ones(2)
    assert ((pareto_reflection(case) == case).all())


def test_raise_init_value_error_with_input_dimension_mismatch():
    nadir, epsilon_avoiding_points, epsilon = np.array([3, 7]), np.array([5, 2]), 1
    with pytest.raises(ValueError, match=r'.*Avoided points must be 2-dimensional array, but is 1-dimensional.*'):
        _ = EpsilonAvoiding(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points, epsilon=epsilon)


def test_raise_value_error_with_differing_input_shapes():
    nadir, epsilon_avoiding_points, epsilon = np.array([3, 7]), np.array([[5, 2], [1, 2]]), 1
    with pytest.raises(ValueError,
                       match=r'.*Dimension mismatch, input vector has dimension 3 but dimension 2 is expected.*'):
        pareto_reflection = EpsilonAvoiding(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points,
                                            epsilon=epsilon)
        case = np.ones(3)
        pareto_reflection(case)

    nadir, epsilon_avoiding_points, epsilon = np.array([3, 7]), np.array([[5, 2, 1], [1, 2, 4]]), 1
    with pytest.raises(ValueError,
                       match=r'.*Dimension mismatch, input vector has dimension 2 but dimension 3 is expected.*'):
        pareto_reflection = EpsilonAvoiding(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points,
                                            epsilon=epsilon)
        case = np.ones(2)
        pareto_reflection(case)
