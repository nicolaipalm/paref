from abc import abstractmethod
from typing import Optional, List

import numpy as np

from paref.interfaces.decorators import initialize_empty_list_of_pareto_reflections, store_pareto_reflections
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

    #TODO: this should be in a contributing.md
    """

    def __init_subclass__(cls):
        """Ensure storing of Pareto reflections in every subclass
        """
        super().__init_subclass__()
        cls.__init__ = initialize_empty_list_of_pareto_reflections(cls.__init__)
        cls.next = store_pareto_reflections(cls.next)

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

    @property
    def used_pareto_reflections(self) -> List[ParetoReflection]:
        """

        Returns
        -------
        List[ParetoReflection]
            List of Pareto reflections used in the sequence in the respective order
        """
        return self._used_pareto_reflections

    def best_fits(self, points: np.ndarray) -> np.ndarray:
        best_fits = []
        for pareto_reflection in self.used_pareto_reflections:
            if pareto_reflection is not None:
                best_fits += pareto_reflection.best_fits(points).tolist()
        return np.unique(best_fits, axis=0)
