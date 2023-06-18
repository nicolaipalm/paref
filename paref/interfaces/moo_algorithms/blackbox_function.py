from abc import abstractmethod
from typing import Union

import numpy as np

from paref.black_box_functions.design_space.bounds import Bounds
from paref.helper_functions.return_pareto_front import return_pareto_front


class BlackboxFunction:
    """Generic interface for blackbox functions used in Paref

    # TODO: finish

    This class provides a generic interface for blackbox functions. In Paref, the evaluations of the blackbox function
    are stored and can be accessed within the BlackboxFunction class.
    In addition, this class stores all the information about the blackbox function

    ..math::
        f:S\to \mathbb{R^d}.

    This consists of the following

    **The design space S:**
    Of which dimension is S?
    How is S defined (f.e. if S is a cube, then, what are the bounds in each dimension)?


    **The assignment f:**
    For a given x in S, what is the value f(x)?

    **The target space:**
    What is the dimension of the target space, i.e. what is d?


    Examples
    --------



    """

    def __init__(self):
        # TODO: bug this is not inherited by implementing functions
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

        Need to be stored in output!
        """
        pass

    @property
    @abstractmethod
    def dimension_design_space(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension_target_space(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def design_space(self) -> Union[Bounds]:
        """
        Currently supports:
        - bounds:

        Returns
        -------

        """
        raise NotImplementedError

    @property
    def evaluations(self):
        return self._evaluations

    @evaluations.setter
    def evaluations(self, evaluations):
        self._evaluations = evaluations

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

    @property
    def pareto_front(self):
        return return_pareto_front(self.y)
