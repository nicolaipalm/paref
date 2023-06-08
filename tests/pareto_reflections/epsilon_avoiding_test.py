import numpy as np

from paref.pareto_reflections.epsilon_avoiding import EpsilonAvoiding


def test_reflecting_example_case():
    nadir, epsilon_avoiding_points, epsilon = np.array([3, 7]), np.array([[2, 1], [1, 5]]), 1
    pareto_reflection = EpsilonAvoiding(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points, epsilon=epsilon)
    assert ((pareto_reflection(np.ones(2)) == nadir).all())
