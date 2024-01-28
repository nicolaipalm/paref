from typing import Callable

import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection


def calculate_optimal_epsilon(g: Callable,
                              bbf: BlackboxFunction) -> float:
    """Calculate optimal epsilon for Pareto reflection minimizing some function g

    If the Pareto reflection takes the form

    .. math::
        p(x) = \sum_{i=1,...,n}\\epsilon x_{i}+g(x)

    then this function calculates the optimal epsilon such that for the current optimum c of g,
    it holds

    .. math::
        0.01g(c) = \\epsilon \sum_{i=1,...,n}c_{i}

    i.e. such that the Pareto reflection is stable.

    Parameters
    ----------
    g : Callable
        function to minimize

    bbf : BlackboxFunction
        blackbox function to which Pareto reflection is applied

    Returns
    -------
    float
        optimal epsilon
    """
    if len(bbf.y) == 0:
        return 1e-3

    c = bbf.y[np.argmin([g(y) for y in bbf.y])]
    if np.abs(np.sum(c)) > 1e-4 and np.abs(g(c)) > 1e-4:
        epsilon = np.abs(0.01 * g(c) / np.sum(c))

    else:
        epsilon = 0.01 * (np.abs(g(c)) + 1) / (np.abs(np.sum(c)) + 1)

    if epsilon < 1e-3:
        epsilon = 1e-3

    return epsilon


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

    def __init__(self,
                 g: Callable,
                 blackbox_function: BlackboxFunction, ):
        self.bbf = blackbox_function
        self._counter = 0
        self.g = g

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._counter <= 2:
            self._epsilon = calculate_optimal_epsilon(self.g, self.bbf)
            self._counter += 1
        return self.g(x) + self._epsilon * np.sum(x)

    @property
    def dimension_domain(self) -> int:
        return self.bbf.dimension_target_space

    @property
    def dimension_codomain(self) -> int:
        return 1
