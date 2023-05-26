import numpy as np

from paref.pareto_reflections.restricting import Restricting
from paref.sequences_pareto_reflections.repeating_sequence import RepeatingSequence
from paref.sequences_pareto_reflections.stopping_criteria.max_iterations_reached import MaxIterationsReached


def test_reflecting_example_case():
    stopping_criteria = MaxIterationsReached(max_iterations=1)
    pareto_reflecting_functions = [Restricting(nadir=np.ones(1), restricting_point=np.ones(1))]
    sequence = RepeatingSequence(pareto_reflecting_functions=pareto_reflecting_functions,
                                 stopping_criteria=stopping_criteria)

    assert (sequence.next().__class__.__name__ == 'Restricting')
    assert (sequence.next() is None)
