from abc import abstractmethod

import numpy as np


class Function:
    """

    """

    def __init__(self):
        self._evaluations = []

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Append [x,f(x)] for self._evaluations
        :param x:
        :type x:
        :return:
        :rtype:

        both, x and output are 2 dimensional arrays
        """
        pass

    @property
    def evaluations(self):
        return self._evaluations

    def clear_evaluations(self):
        self._evaluations = []
