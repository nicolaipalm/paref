from abc import abstractmethod
from typing import Optional
from paref.interfaces.pareto_reflections.pareto_reflecting_function import ParetoReflectingFunction


class SequenceParetoReflectingFunctions:
    """Interface for sequences of Pareto reflections

    Documentation of an implementation of this interface should contain:

    When to use
    -----------
    This Pareto reflection should be used if...

    What it does
    ------------
    The Pareto points of this map are...

    Mathematical formula
    --------------------

    Examples
    --------

    #TODO: this should be in a contributing.md
    """

    @abstractmethod
    def next(self, ) -> Optional[ParetoReflectingFunction]:
        """

        Returns
        -------
        Optional[ParetoReflectingFunction]
            Either the next Pareto reflection or None if the end of the sequence is reached
        """
        pass
