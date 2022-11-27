import numpy as np

from weimoo.pareto_reflecting_library.functions.interfaces.pareto_reflecting_function import ParetoReflectingFunction


class WeightedNormToUtopia(ParetoReflectingFunction):
    def __init__(self, utopia_point: np.ndarray, potency: np.ndarray, scalar: np.ndarray):
        self.potency = potency
        self.scalar = scalar
        self.utopia_point = utopia_point

    def __call__(self, x: np.ndarray):
        return np.sum(self.scalar * ((x-self.utopia_point) ** self.potency))
