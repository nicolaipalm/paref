from abc import abstractmethod
from typing import Union, List

import numpy as np

from paref.black_box_functions.design_space.bounds import Bounds


class BlackboxFunction:
    """Generic interface for blackbox functions used in Paref

    This class provides a generic interface for blackbox functions. In Paref, the evaluations of the blackbox function
    need to be stored and can then be accessed within the BlackboxFunction class.
    In addition, this class stores all the information about the blackbox function

    .. math::

        f:S \\to \mathbb{R}^d.

    This consists of the following

    **The design space S:**
    Of which dimension is S?
    How is S defined?
    Currently, this supports:

    * cubes (characterized be its bounds), i.e.

    .. math::

         S=\\prod_{i=1}^n[a_i,b_i] \\subset \\mathbb{R}^n

    **The target space:**
    What is the dimension of the target space, i.e. what is d?

    **The assignment f:**
    For a given x in S, what is the value f(x)?


    Examples
    --------
    Lets say the blackbox function has the following mathematical expression

    .. math::

        f:[0,1]\\times [0,1]\\to \mathbb{R}^2,f(x)=(x_2^2,x_1-x_2)

    Then, the pythonic blackbox function will be implemented as follows
    # TBA: example

    .. note::

        In most cases, the closed mathematical blackbox functions are not known. Instead,
        it can only be evaluated at a given input. In that case "evaluating" is
        implemented in the

        .. code-block:: python

            __call__(self, x: np.ndarray) -> np.ndarray

        method below.


    """

    def __init__(self):
        """Initialize storage for evaluations of the blackbox function
        """
        self._evaluations = []

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply blackbox function to input and store the tuple (input,output) in self._evaluations

        .. warning::
            When blackbox function is called the list of input and output, i.e. [x,f(x)] must be
            appended  to  self._evaluations!

        Parameters
        ----------
        x : np.ndarray
            input to which the blackbox function is applied

        Returns
        -------
        np.ndarray
            output of blackbox function applied to input


        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension_design_space(self) -> int:
        """

        Returns
        -------
        int
            dimension of design space

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension_target_space(self) -> int:
        """

        Returns
        -------
        int
            dimension of target space

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def design_space(self) -> Union[Bounds]:
        """Characterization of design space

        Currently, this supports:

        * cubes (characterized be its bounds), i.e.

        .. math::

             S=\\prod_{i=1}^n[a_i,b_i] \\subset \\mathbb{R}^n

        Returns
        -------
        Union[Bounds]
            pythonic representation of design space

        """
        raise NotImplementedError

    @property
    def evaluations(self) -> List:
        """

        Returns
        -------
        List
            list of evaluations: each element of the form [input,value of blackbox function at input]


        """
        return self._evaluations

    @evaluations.setter
    def evaluations(self, evaluations: List):
        """Set list of evaluations

        .. warning::
            each element of the list of evaluations must be
            of the form [input,value of blackbox function at input]!

        Parameters
        ----------
        evaluations : List
            set the list of evaluations

        """
        self._evaluations = evaluations

    @property
    def x(self) -> np.ndarray:
        """Numpy array of inputs of all evaluations

        Returns
        -------
        np.ndarray
            array of inputs of all evaluations

        """
        return np.array([evaluation[0] for evaluation in self._evaluations])

    @x.setter
    def x(self, value):
        # TBA: needed?
        for index, evaluation in enumerate(self._evaluations):
            evaluation[0] = value[index]

    @property
    def y(self) -> np.ndarray:
        """Numpy array of outputs of all evaluations

        Returns
        -------
        np.ndarray
            array of outputs of all evaluations

        """
        return np.array([evaluation[1] for evaluation in self._evaluations])

    @y.setter
    def y(self, value):
        # TBA: needed?
        for index, evaluation in enumerate(self._evaluations):
            evaluation[1] = value[index]

    def clear_evaluations(self) -> None:
        """Clear all evaluations

        I.e. set self._evaluations to empty list.
        """
        self._evaluations = []

    @property
    def pareto_front(self) -> np.ndarray:
        """Return Pareto front of evaluation

        Returns
        -------
        np.ndarray
            Pareto front of evaluations

        """
        array = self.y
        pareto_points = []
        for i, point in enumerate(array):
            is_pareto = True
            for j, other in enumerate(array):
                if i == j:
                    continue
                if np.all(point >= other) and np.any(point > other):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_points.append(point)
        return np.array(pareto_points)
