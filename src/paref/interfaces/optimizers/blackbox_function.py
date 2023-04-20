from abc import abstractmethod

import numpy as np


class BlackboxFunction:
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

    @property
    def x(self):
        return np.array([evaluation[0] for evaluation in self._evaluations])

    @x.setter
    def x(self, value):
        for index, evaluation in enumerate(self._evaluations):
            evaluation[0] = value[index]

    @property
    def y(self):
        return np.array([evaluation[1] for evaluation in self._evaluations])

    @y.setter
    def y(self, value):
        for index, evaluation in enumerate(self._evaluations):
            evaluation[1] = value[index]

    def clear_evaluations(self):
        self._evaluations = []
