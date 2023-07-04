import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflection_sequences.generic.next_when_stopping_criteria_met import NextWhenStoppingCriteriaMet
from paref.pareto_reflections.restrict_by_point import RestrictByPoint


def test_repeating_sequence_example_case():
    stopping_criteria = MaxIterationsReached(max_iterations=1)
    pareto_reflecting_functions = [RestrictByPoint(nadir=np.ones(1), restricting_point=np.ones(1))]
    sequence = NextWhenStoppingCriteriaMet(pareto_reflections=pareto_reflecting_functions,
                                           stopping_criteria=stopping_criteria)
    assert (sequence.next(BlackboxFunction()).__class__.__name__ == 'RestrictByPoint')
    assert (sequence.next(BlackboxFunction()) is None)
