from typing import List

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import \
    SequenceParetoReflections
from paref.interfaces.moo_algorithms.stopping_criteria import StoppingCriteria


class RepeatingSequence(SequenceParetoReflections):
    """Define a sequence by repeating Pareto reflections until a stopping criteria is met

    When to use
    -----------
    This sequence should be used if you search for finitely many Pareto points with certain properties
    reflected by a suitable Pareto reflection.
    Notice that by repeating the Pareto reflections you iteratively search for the optimum
    (similar to iteratively minimizing a certain blackbox_function).

    What it does
    ------------
    The sequence simply loops through given Pareto reflections until a defined stopping criteria is met.
    If the end is reached but

    Examples
    --------
    Initialize list of Pareto reflections

    >>> import numpy as np
    >>> from paref.pareto_reflections.restricting import RestrictByPoint
    >>> from paref.pareto_reflection_sequences.stopping_criteria.max_iterations_reached import MaxIterationsReached

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
        """Specify the stopping criteria and the Pareto reflections to be repeated

        # TODO: add by default simply go once through list

        Parameters
        ----------

        stopping_criteria : StoppingCriteria
            stopping criteria defining the end of the sequence

        pareto_reflections : List[ParetoReflection]
            Pareto reflections to be repeated stored in a list
        """
        self._pareto_reflecting_functions = pareto_reflections
        self._iter = -1

    def next(self, blackbox_function: BlackboxFunction) -> ParetoReflection:
        """Return the next Pareto reflection of the sequence

        Returns
        -------
        ParetoReflection
            next Pareto reflection of the sequence

        """
        self._iter += 1
        return self._pareto_reflecting_functions[self._iter % len(self._pareto_reflecting_functions)]
