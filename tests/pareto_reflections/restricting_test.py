import numpy as np

from paref.pareto_reflections.restricting import Restricting


def test_reflecting():
    nadir, restricting_point = np.array([3, 7]), np.zeros(2)
    pareto_reflection = Restricting(nadir=nadir, restricting_point=restricting_point)
    assert ((pareto_reflection(np.ones(2)) == np.array([3, 7])).all())
