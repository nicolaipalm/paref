import numpy as np

from src.interfaces.weight_function import WeightFunction


class Potency(WeightFunction):
    def __init__(self, potency: np.ndarray):
        self.potency = potency

    def __call__(self, x):
        return np.sum(x ** self.potency)
