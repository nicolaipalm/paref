import numpy as np

from src.interfaces.weight_function import WeightFunction


class ScalarPotency(WeightFunction):
    def __init__(self, potency: np.ndarray, scalar: np.ndarray):
        self.potency = potency
        self.scalar = scalar

    def __call__(self, x):
        return np.sum(self.scalar * (x ** self.potency))
