import numpy as np

from src.interfaces.function import Function
from src.interfaces.minimizer import Minimizer
from src.interfaces.weight_function import WeightFunction


class WeightBasedMOO:
    def __init__(self, weight_function: WeightFunction):
        self._weight_function = weight_function

    def __call__(self,
                 function: Function,
                 minimizer: Minimizer,
                 max_evaluations: int,
                 upper_bounds: np.ndarray,
                 lower_bounds: np.ndarray) -> np.ndarray:
        function.clear_evaluations()
        return minimizer(function=lambda x: self._weight_function(function(x)),
                         max_iter=max_evaluations,
                         upper_bounds=upper_bounds,
                         lower_bounds=lower_bounds)
