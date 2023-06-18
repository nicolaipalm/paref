from abc import abstractmethod
from typing import Optional

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection


class SequenceParetoReflections:
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
    def next(self, blackbox_function: BlackboxFunction) -> Optional[ParetoReflection]:
        """

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            The blackbox function to which the algorithm is applied

        Returns
        -------
        Optional[ParetoReflection]
            Either the next Pareto reflection or None if the end of the sequence is reached
        """
        pass
