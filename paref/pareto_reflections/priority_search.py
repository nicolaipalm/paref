from typing import Callable

import numpy as np
import scipy as sp

from paref.pareto_reflections.find_edge_points import FindEdgePoints
from paref.pareto_reflections.minimize_g import MinGParetoReflection


class PrioritySearch(MinGParetoReflection):
    """Find Pareto points which reflect your priorities

    When to use
    -----------
    This Pareto reflection should be used if you wish to obtain Pareto points where different weights are assigned to
    the different dimensions of the target space.

    .. warning::

            This Pareto reflection assumes that an edge point corresponding to each component
            was already (approximately) found.
            Use the ``FindEdgePointsSequence`` sequence of Pareto reflections first.

    """

    def __init__(self, blackbox_function, priority: np.ndarray):
        super().__init__(blackbox_function=blackbox_function)
        if len(priority) != blackbox_function.dimension_target_space:
            raise ValueError(f'Priority vector has dimension {len(priority)} but must have dimension'
                             f'{blackbox_function.dimension_target_space}!')

        priority = -(priority - np.sum(priority))
        priority = np.array(priority) / np.sum(priority)

        self._minima_components_pp = []
        # TODO: it might happen that the edge points collapse i.e. that two edge points agree. than this algo fails
        for i in range(len(priority)):
            self._minima_components_pp.append(
                FindEdgePoints(dimension=i, blackbox_function=blackbox_function).best_fits(
                    blackbox_function.y)[0])

        self._center = np.sum(
            [priority[i] * self._minima_components_pp[i] for i in range(blackbox_function.dimension_target_space)],
            axis=0)
        # calculate orthogonal basis
        span = self._minima_components_pp - self._center
        self.orthogonal_basis = sp.linalg.orth(span.T).T

    @property
    def g(self) -> Callable:
        return lambda x: np.linalg.norm(np.sum(np.array([np.dot(x - self._center, basis_vector) * basis_vector
                                                         for basis_vector in self.orthogonal_basis]), axis=0))
