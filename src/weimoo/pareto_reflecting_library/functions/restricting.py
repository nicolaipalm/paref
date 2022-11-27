import numpy as np

from weimoo.pareto_reflecting_library.functions.interfaces.pareto_reflecting_function import ParetoReflectingFunction


class Restricting(ParetoReflectingFunction):
    def __init__(self, nadir: np.ndarray,
                 restricting_point: np.ndarray, ):
        self.nadir = nadir
        self.restricting_point = restricting_point

    def __call__(self, x: np.ndarray):
        if x.shape != self.restricting_point.shape:
            raise ValueError(
                f"Shapes don't match! Shape of x is {x.shape}, shape of restricting point is {self.restricting_point.shape}!")

        if np.any(self.restricting_point <= x):
            return self.nadir

        return x
