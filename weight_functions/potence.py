import numpy as np

from interfaces.weight_function import WeightFunction


class Potence(WeightFunction):
    def __init__(self, potence: np.ndarray):
        self.potence = potence

    def __call__(self, x):
        return np.sum(x ** self.potence)
