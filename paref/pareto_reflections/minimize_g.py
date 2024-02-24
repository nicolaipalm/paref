from abc import abstractmethod
from typing import Callable, Union

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

        p(x) = \sum_{i=1,...,n}\\epsilon scaling_x(x)_{i}+scaling_g(g(x))


    """

    def __init__(self,
                 blackbox_function: BlackboxFunction,
                 epsilon: Union[float, np.ndarray] = 1e-2,
                 scaling_g: Callable[[np.ndarray], np.ndarray] = lambda x: x,
                 scaling_x: Callable[[np.ndarray], np.ndarray] = lambda x: x, ):
        """

        Parameters
        ----------

        blackbox_function : BlackboxFunction
            blackbox function to which Pareto reflection is applied

        epsilon : Union[float, np.ndarray] default 1e-3
            epsilon determining weight of components

        scaling_g : Callable[[np.ndarray], np.ndarray] default lambda x: x
            scaling function for g

        scaling_x : Callable[[np.ndarray], np.ndarray] default lambda x: x
            scaling function for x
        """
        self.bbf = blackbox_function
        self._epsilon = epsilon
        self._counter = 0
        self.scaling_g = scaling_g
        self.scaling_x = scaling_x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.scaling_g(self.g(x)) + np.sum(self._epsilon * self.scaling_x(x))

    @property
    def dimension_domain(self) -> int:
        return self.bbf.dimension_target_space

    @property
    def dimension_codomain(self) -> int:
        return 1

    @property
    @abstractmethod
    def g(self) -> Callable[[np.ndarray], np.ndarray]:
        raise NotImplementedError
