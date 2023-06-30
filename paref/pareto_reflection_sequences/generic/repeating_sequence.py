from typing import List

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import \
    SequenceParetoReflections


class RepeatingSequence(SequenceParetoReflections):
    """Define a sequence by repeating a list of Pareto reflections

    This is probably the most generic way to define a sequence of Pareto reflections:
    specify a (finite) list of Pareto reflections and repeat that list.
    It can be seen as implementing a list of Pareto reflections (which is a sequence) in the
    sequence of Pareto reflections interface.

    When to use
    -----------
    # TBA: when to use?

    What it does
    ------------
    The sequence simply loops through given list of Pareto reflections
    If the end is reached, the sequence starts again from the beginning of the list of reflections.

    Examples
    --------
    # TBA: meaningful example when to use this
    Initialize list of Pareto reflections

    >>> import numpy as np
    >>> from paref.pareto_reflections.restrict_by_point import RestrictByPoint
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
    Restricting

    >>> sequence.next()
    None
    """

    def __init__(self,
                 pareto_reflections: List[ParetoReflection],
                 ):
        """Specify the list of Pareto reflections you want to loop through

        Parameters
        ----------
        pareto_reflections : List[ParetoReflection]
            Pareto reflections to be repeated stored in a list
        """
        self._pareto_reflecting_functions = pareto_reflections
        self._iter = -1

    def next(self, blackbox_function: BlackboxFunction) -> ParetoReflection:
        """Return the next Pareto reflection of the list of Pareto reflections

        Returns
        -------
        ParetoReflection
            successor Pareto reflection of the sequence

        """
        self._iter += 1
        return self._pareto_reflecting_functions[self._iter % len(self._pareto_reflecting_functions)]
