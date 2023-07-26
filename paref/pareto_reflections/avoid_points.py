import numbers
from typing import Union

import numpy as np

from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection


class AvoidPoints(ParetoReflection):
    """Avoid certain areas/points (Pareto reflection)

        When to use
        -----------
        This Pareto reflection should be used if Pareto points are desired which have a minimum distance to some points
        in each component

        What it does
        ------------
        The Pareto points of this map are all the Pareto points which have distance at least epsilon (to be specified)
        in each component

        Mathematical formula
        --------------------
        Denote by D the set of points which should be avoided, let epsilon be some positive real number
        and n some nadir (i.e. n dominated by all points). Then,

        .. math::
            p(x) = n

        if x plus epsilon is dominated or equal to d for some d in D and

        .. math::
            p(x) = x

        else.
        Notice that each so found Pareto point has distance at least epsilon in each component from every point in D.

        Examples
        --------
        Define the nadir and the points which should be avoided

        >>> import numpy as np
        >>> nadir, epsilon_avoiding_points, epsilon = np.array([3,7]), np.array([[2,1],[1,5]]), 1

        Initialize the EpsilonAvoiding

        >>> pareto_reflection = AvoidPoints(nadir=nadir, epsilon_avoiding_points=epsilon_avoiding_points, \
            epsilon=epsilon)

        Calling it to (1,1), i.e. calculating

        .. math::
            p(1,1)=(3,7)

        since (1,1) is dominated by (2,1)-1
        yields:

        >>> pareto_reflection(np.ones(2))
        array([3, 7])

        """

    def __init__(self,
                 nadir: np.ndarray,
                 epsilon_avoiding_points: np.ndarray,
                 epsilon: Union[numbers.Real, np.ndarray]):
        """Specify the nadir and the to be avoided points

        Parameters
        ----------
        nadir : np.ndarray
            nadir (dominated by all points) stored in 1 dimensional array of length n

        epsilon_avoiding_points : np.ndarray
            avoided points stored in 2-dimensional array with first dimension corresponding to the points

        epsilon : Union[numbers.Real, np.ndarray])
            value which is subtracted from avoided points
        """

        if nadir.shape != epsilon_avoiding_points.shape and nadir.shape != epsilon_avoiding_points[0].shape:
            raise ValueError('Nadir and avoiding points need to be 2-dimensional numpy arrays of equal shape!')

        # TBA: error handling rest

        if not isinstance(epsilon, numbers.Real) and epsilon.shape != nadir.shape:
            raise ValueError('Epsilon must be a Real Number or a numpy array of same shape as nadir!')

        if np.any(epsilon < 0):
            raise ValueError('Epsilon must be positive!')

        self.nadir = nadir
        self.epsilon = epsilon
        self.epsilon_avoiding_points = epsilon_avoiding_points

    def __call__(self, x: np.ndarray):
        """Calculate the epsilon avoiding function

        Parameters
        ----------
        x :
        np.ndarray
            input vector stored in 1 dimensional array of length n

        Returns
        -------
        float
            value of the epsilon avoiding function

        """
        if len(x.shape) != 1:
            raise ValueError(f'Input x must be of dimension 1! Shape of x is {x.shape}.')
        for point in self.epsilon_avoiding_points:
            if np.all(point - self.epsilon <= x):
                return self.nadir

        return x

    @property
    def dimension_codomain(self) -> int:
        return len(self.nadir)

    @property
    def dimension_domain(self) -> int:
        return len(self.nadir)
