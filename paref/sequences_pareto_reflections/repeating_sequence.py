from typing import List

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflecting_function import ParetoReflectingFunction
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflecting_functions import \
    SequenceParetoReflectingFunctions
from paref.interfaces.optimizers.stopping_criteria import StoppingCriteria


class RepeatingSequence(SequenceParetoReflectingFunctions):
    """Define a sequence by repeating Pareto reflections until a stopping criteria is met

    When to use
    -----------
    This sequence should be used if you search for finitely many Pareto points with certain properties
    reflected by a suitable Pareto reflection.
    Notice that by repeating the Pareto reflections you iteratively search for the optimum
    (similar to iteratively minimizing a certain function).

    What it does
    ------------
    The sequence simply loops through given Pareto reflections until a defined stopping criteria is met.
    If the end is reached but

    Examples
    --------
    Initialize list of Pareto reflections

    >>> import numpy as np
    >>> from paref.pareto_reflections.restricting import Restricting
    >>> pareto_reflecting_functions = [Restricting(nadir=np.ones(1),restricting_point=np.ones(1))]

    Initialze stopping criteria and blackbox function
    #TODO: fill both
    >>>
    >>> stopping_criteria =

    Initialize repeating sequence

    >>> sequence = RepeatingSequence(stopping_criteria=stopping_criteria, \
                                     pareto_reflecting_functions=pareto_reflecting_functions)

    The repeating sequence returns the given Pareto reflection in each step of iteration until the stopping criteria is met
    >>> sequence.next(blackbox_function=blackbox_function)
    #TODO: fill output
    >>> sequence.next(blackbox_function=blackbox_function)
    None
    """

    def __init__(self,
                 stopping_criteria: StoppingCriteria,
                 pareto_reflecting_functions: List[ParetoReflectingFunction]):
        """Specify the stopping criteria and the Pareto reflections to be repeated

        Parameters
        ----------
        stopping_criteria :
        StoppingCriteria
            stopping criteria defining the end of the sequence

        pareto_reflecting_functions :
        List[ParetoReflectingFunction]
            Pareto reflections to be repeated stored in a list
        """
        self._stopping_criteria = stopping_criteria
        self._pareto_reflecting_functions = pareto_reflecting_functions
        self._iter = 0

    def next(self,
             blackbox_function: BlackboxFunction
             ) -> ParetoReflectingFunction:
        """Return the next Pareto reflection of the sequence

        Parameters
        ----------
        blackbox_function :
        BlackboxFunction
            blackbox function to be optimized

        Returns
        -------
        ParetoReflectingFunction
            next Pareto reflection of the sequence

        """
        # TODO: What to do if the end of the sequence is reached?
        if self._stopping_criteria(blackbox_function):
            self._iter += 1
            return self._pareto_reflecting_functions[self._iter % len(self._pareto_reflecting_functions)]

        else:
            raise ValueError("Stopping criteria is reached")
