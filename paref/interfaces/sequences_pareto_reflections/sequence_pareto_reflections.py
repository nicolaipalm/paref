from abc import abstractmethod
from typing import Optional

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection


class SequenceParetoReflections:
    """Interface for pareto_reflections of Pareto reflections

    A sequence of Pareto reflections is a mathematical sequence

    .. math::

        (p_i)_{i \\in \mathbb{N}}

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

    #TBA: this should be in a contributing.md
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
