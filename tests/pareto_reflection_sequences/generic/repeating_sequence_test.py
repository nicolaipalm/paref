import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.pareto_reflection_sequences.generic.repeating_sequence import RepeatingSequence
from paref.pareto_reflections.restrict_by_point import RestrictByPoint


def test_repeating_sequence_example_case():
    pareto_reflecting_functions = [RestrictByPoint(nadir=np.ones(1), restricting_point=np.ones(1))]
    sequence = RepeatingSequence(pareto_reflections=pareto_reflecting_functions)

    assert (sequence.next(BlackboxFunction()).__class__.__name__ == 'RestrictByPoint')
    assert (sequence.next(BlackboxFunction()).__class__.__name__ == 'RestrictByPoint')
