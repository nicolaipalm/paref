import warnings
from abc import abstractmethod
from typing import Union, List

import numpy as np
from scipy.stats import qmc

from paref.blackbox_functions.design_space.bounds import Bounds
from paref.interfaces.decorators import initialize_empty_evaluations, store_evaluation_bbf


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

    >>> class BlackboxFunctionExample(BlackboxFunction):
    >>>     def __call__(self, x: np.ndarray) -> np.ndarray:
    >>>         return np.array([x[1] ** 2, x[0] - x[1]])
    >>>
    >>>     @property
    >>>     def dimension_design_space(self) -> int:
    >>>         return 2
    >>>
    >>>     @property
    >>>     def dimension_target_space(self) -> int:
    >>>         return 2

    """

    def __init_subclass__(cls):
        """Ensure storing of evaluations in every subclass
        and initialize empty evaluations list in subclasses
        """
        super().__init_subclass__()
        cls.__init__ = initialize_empty_evaluations(cls.__init__)
        cls.__call__ = store_evaluation_bbf(cls.__call__)

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply blackbox function to input and store the tuple (input,output) in self._evaluations

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
        np.ndarray
            numpy array of evaluations: each element of the form [input,value of blackbox function at input]


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

    def perform_lhc(self, n: int) -> None:
        """Perform Latin Hypercube Sampling

        The `latin hypercube sampling <https://en.wikipedia.org/wiki/Latin_hypercube_samplingL>`_ (LHC) is
        a stratified (random) sampling method.
        It is often used as a powerful initial sampling method in order to explore the design space.


        Parameters
        ----------
        n : int
            number of LHC samples

        """
        previous_number_evaluations = len(self.evaluations)
        if isinstance(self.design_space, Bounds):
            [self(x) for x in qmc.scale(
                qmc.LatinHypercube(d=self.dimension_design_space).random(
                    n=n),
                self.design_space.lower_bounds,
                self.design_space.upper_bounds,
            )]  # add samples according to latin hypercube scheme

        else:
            raise ValueError('Design space property of blackbox function must be an instance of Bounds!')

        if previous_number_evaluations != len(self.evaluations) - n:
            warnings.warn(f'It seems like you are storing too less or many evaluations. '
                          f'Number of evaluations before LHC is {previous_number_evaluations} '
                          f'and afterwards {len(self.evaluations)}.')

    def save(self, path: str) -> None:
        """Save evaluations to npy-file

        For each evaluation, the input and output are concatenated and stored in a row of the npy-file.

        Parameters
        ----------
        path : str
            path to file

        """
        np.save(path, np.concatenate((self.x, self.y), axis=1))

    def load(self, path: str) -> None:
        """Load evaluations from npy-file

        Parameters
        ----------
        path : str
            path to file

        """
        evals = np.load(path)
        if evals.shape[1] != self.dimension_design_space + self.dimension_target_space:
            raise ValueError(f'Loaded evaluations do not match target resp. design space dimension'
                             f'({self.dimension_design_space} resp. {self.dimension_target_space})!')

        else:
            self.evaluations = [[evaluation[:self.dimension_design_space], evaluation[self.dimension_design_space:]] for
                                evaluation in evals]

    @property
    def pareto_front(self) -> np.ndarray:
        """Return Pareto front of evaluation

        Returns
        -------
        np.ndarray
            Pareto front of target vectors of evaluations

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
