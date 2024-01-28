import numpy as np
import scipy as sp

from paref.pareto_reflections.find_1_pareto_points import Find1ParetoPoints
from paref.pareto_reflections.minimize_g import MinGParetoReflection


class PrioritySearch(MinGParetoReflection):
    def __init__(self, blackbox_function, priority: np.ndarray):
        self._counter = 0
        self.bbf = blackbox_function
        if len(priority) != blackbox_function.dimension_target_space:
            raise ValueError(f'Priority vector has dimension {len(priority)} but must have dimension'
                             f'{blackbox_function.dimension_target_space}!')

        priority = np.array(priority) / np.sum(priority)

        self._minima_components_pp = []
        for i in range(len(priority)):
            self._minima_components_pp.append(
                Find1ParetoPoints(dimension=i, blackbox_function=blackbox_function).best_fits(
                    blackbox_function.y)[0])

        self._center = np.sum(
            [priority[i] * self._minima_components_pp[i] for i in range(blackbox_function.dimension_target_space)],
            axis=0)
        print(self._minima_components_pp, self._center, priority)

        # calculate orthogonal basis
        span = self._minima_components_pp - self._center
        self.orthogonal_basis = sp.linalg.orth(span.T).T
        self.g = lambda x: np.linalg.norm(np.sum(np.array([np.dot(x - self._center, basis_vector) * basis_vector
                                                           for basis_vector in self.orthogonal_basis]), axis=0))
