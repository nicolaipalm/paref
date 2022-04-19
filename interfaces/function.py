import numpy as np


class Function:
    """

    """

    def __init__(self):
        self._evaluations = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Append [x,f(x)] for self._evaluations
        :param x:
        :type x:
        :return:
        :rtype:
        """
        pass

    def partial_derivative(self, x: np.ndarray, partial_derivative_index: int) -> np.ndarray:
        pass

    def second_partial_derivative(self, x: np.ndarray, partial_derivative_index_1: int,
                                  partial_derivative_index_2: int) -> np.ndarray:
        pass

    @property
    def evaluations(self):
        return self._evaluations
