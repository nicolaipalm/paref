import numpy as np

from paref.pareto_reflections.minimize_weighted_norm_to_utopia import MinimizeWeightedNormToUtopia


def test_reflecting_example_case():
    utopia_point, potency, scalar = np.zeros(2), np.array([2]), np.ones(2)
    pareto_reflection = MinimizeWeightedNormToUtopia(utopia_point=utopia_point, potency=potency, scalar=scalar)
    assert ((pareto_reflection(np.ones(2)) == np.array([2**(1/2)])).all())
