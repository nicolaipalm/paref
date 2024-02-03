from abc import abstractmethod
from typing import Callable

import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection


class MinGParetoReflection(ParetoReflection):
    """Find a Pareto point among all points minimizing some function g

     When to use
    -----------
    This Pareto reflection should be used if a Pareto point is desired which is Pareto optimal
    among all points minimizing some function g.

    .. warning::

        This Reflection is not necessarily Pareto reflecting. It is only if the points which are Pareto optimal
        among all points that minimize g are in fact Pareto optimal.


    Mathematical formula
    --------------------

    .. math::

        p(x) = \sum_{i=1,...,n}\\epsilon x_{i}+g(x)


    """

    def __init__(self,
                 blackbox_function: BlackboxFunction,
                 epsilon: float = 1e-3):
        """

        Parameters
        ----------
        g :
            function which is minimized

        blackbox_function :
            blackbox function to which Pareto reflection is applied

        epsilon :
            epsilon determining weight of components
        """
        self.bbf = blackbox_function
        self._epsilon = epsilon
        self._counter = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # ensure that resulting Pareto reflection is robust against translations
        if self._counter <= 2:
            self._c = np.min(self.bbf.y, axis=0)
            self._k = np.min([self.g(y) for y in self.bbf.y])
            self._mg = np.min([self.g(y) for y in self.bbf.y])
            self._counter += 1
        return self.g(x) - self._k + np.sum(self._epsilon * (x - self._c))

    @property
    def dimension_domain(self) -> int:
        return self.bbf.dimension_target_space

    @property
    def dimension_codomain(self) -> int:
        return 1

    @property
    @abstractmethod
    def g(self) -> Callable:
        raise NotImplementedError
