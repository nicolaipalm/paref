import numpy as np

from paref.pareto_reflections.restrict_by_point import RestrictByPoint
from paref.pareto_reflection_sequences.generic.repeating_sequence import RepeatingSequence
from paref.moo_algorithms.stopping_criteria import MaxIterationsReached


def test_reflecting_example_case():
    stopping_criteria = MaxIterationsReached(max_iterations=1)
    pareto_reflecting_functions = [RestrictByPoint(nadir=np.ones(1), restricting_point=np.ones(1))]
    sequence = RepeatingSequence(pareto_reflections=pareto_reflecting_functions)

    assert (sequence.next().__class__.__name__ == 'RestrictByPoint')
    assert (sequence.next() is None)
