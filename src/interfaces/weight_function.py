import numpy as np


class WeightFunction:
    """
    The weight function transforms multiple dimensional input into scalars.
    It satisfies that a minimum of the weight function is a pareto optimum (but n.n. vice versa).
    """

    def __call__(self, x: np.ndarray) -> float:
        pass

    def partial_derivative(self, x: np.ndarray, partial_derivative_index: int) -> float:
        pass

    def second_partial_derivative(self, x: np.ndarray, partial_derivative_index_1: int,
                                  partial_derivative_index_2: int) -> float:
        pass
