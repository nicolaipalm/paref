import numpy as np

from paref.pareto_reflections.weighted_norm_to_utopia import WeightedNormToUtopia


def test_reflecting():
    utopia_point, potency, scalar = np.zeros(2), np.array([2]), np.ones(2)
    pareto_reflection = WeightedNormToUtopia(utopia_point=utopia_point, potency=potency, scalar=scalar)
    assert ((pareto_reflection(np.ones(2)) == np.array([2])).all())
