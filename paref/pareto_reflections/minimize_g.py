from typing import Callable

import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection


class MinGParetoReflection(ParetoReflection):
    """Find a Pareto point among all points minimizing some function g

     When to use
    -----------
    This Pareto reflection should be used if a Pareto point is desired which minimizes some function g.

    Mathematical formula
    --------------------

    .. math::
        p(x) = \sum_{i=1,...,n}\\epsilon x_{i}+g(x)


    """
    def __init__(self, g: Callable, bbf: BlackboxFunction):
        self.g = g
        self.bbf = bbf

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # TODO: automatically set optimal epsilon
        epsilon = 1e-3
        return self.g(x) + epsilon * np.sum(x)

    @property
    def dimension_domain(self) -> int:
        return self.bbf.dimension_target_space

    @property
    def dimension_codomain(self) -> int:
        return 1
