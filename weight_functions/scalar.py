import numpy as np

from interfaces.weight_function import WeightFunction


class Scalar(WeightFunction):
    def __init__(self, theta: np.ndarray):
        self.theta = theta

    def __call__(self, x):
        return np.sum(x * self.theta)
