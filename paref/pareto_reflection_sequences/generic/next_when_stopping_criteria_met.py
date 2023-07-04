from typing import List, Optional

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import \
    SequenceParetoReflections
from paref.interfaces.moo_algorithms.stopping_criteria import StoppingCriteria


class NextWhenStoppingCriteriaMet(SequenceParetoReflections):
    """Define a sequence by moving on to the next Pareto reflection if convergence is reached

    When to use
    -----------
    This sequence should be used if you want to repeat a single Pareto reflection until the Pareto point it
    is looking for is sufficiently close approximated (to be specified in a stopping criteria).

    What it does
    ------------
    The sequence simply loops through given Pareto reflections until a defined stopping criteria is met.
    If the end is reached, this sequence returns None, indicating the end of the sequence.

    Examples
    --------
    # TBA: add
    Initialize list of Pareto reflections

    >>> import numpy as np
    >>> from paref.pareto_reflections.restrict_by_point import RestrictByPoint
    >>> from paref.pareto_reflection_sequences.generic.repeating_sequence import RepeatingSequence
    >>> from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached

    Initialze stopping criteria

    >>> stopping_criteria = MaxIterationsReached(max_iterations=1)

    Initialize Pareto reflection to be repeated

    >>> pareto_reflecting_functions = [RestrictByPoint(nadir=np.ones(1),restricting_point=np.ones(1))]

    Initialize repeating sequence

    >>> sequence = RepeatingSequence(pareto_reflections=pareto_reflecting_functions)

    The repeating sequence returns the given Pareto reflection in each step of iteration until the stopping criteria is
    met

    >>> sequence.next().__class__.__name__
    RestrictByPoint

    >>> sequence.next()
    None
    """

    def __init__(self,
                 pareto_reflections: List[ParetoReflection],
                 stopping_criteria: StoppingCriteria,
                 ):
        """Specify the stopping criteria and the Pareto reflections to be repeated

        Parameters
        ----------

        stopping_criteria : StoppingCriteria
            stopping criteria indicating when to move to the next Pareto reflection

        pareto_reflections : List[ParetoReflection]
            Pareto reflections stored in a list
        """
        self._pareto_reflections = pareto_reflections
        self._iter = 0
        self._stopping_criteria = stopping_criteria

    def next(self, blackbox_function: BlackboxFunction) -> Optional[ParetoReflection]:
        """Return the next Pareto reflection of the sequence if stopping criteria is met

        Returns
        -------
        ParetoReflection
            next Pareto reflection of the list or same Pareto reflection as before if stopping criteria is not met

        """
        if self._stopping_criteria(blackbox_function):
            self._iter += 1

        if self._iter == len(self._pareto_reflections):
            return None

        return self._pareto_reflections[self._iter]
