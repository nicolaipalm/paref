from typing import Union

import numpy as np

from paref.interfaces.pareto_reflections.pareto_reflecting_function import ParetoReflectingFunction


class EpsilonAvoiding(ParetoReflectingFunction):
    def __init__(self,
                 nadir: np.ndarray,
                 epsilon_avoiding_points: np.ndarray,
                 epsilon: Union[float, np.ndarray] ):
        self.nadir = nadir
        self.epsilon = epsilon
        self.epsilon_avoiding_points = epsilon_avoiding_points

    def __call__(self, x: np.ndarray):
        for _, point in enumerate(self.epsilon_avoiding_points):
            if np.all(point - self.epsilon <= x):
                return self.nadir

        return x
