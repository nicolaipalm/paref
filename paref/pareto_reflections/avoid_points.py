from typing import Union

import numpy as np

from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection


class AvoidPoints(ParetoReflection):
    """Avoid certain areas/points (Pareto reflection)

        When to use
        -----------
        This Pareto reflection should be used if Pareto points are desired which are not dominated by certain points
        (plus some epsilon)

        What it does
        ------------
        The Pareto points of this map are all the Pareto points which are not dominated by that point(s)
        (minus some epsilon)

        Mathematical formula
        --------------------
        Denote by D the set of points which should not dominate, let epsilon be some strictly positive real number
        and n some nadir (i.e. n dominated by all points). Then,

        .. math::
            p(x) = n

        if x plus epsilon is dominated or equal to d for some d in D and

        .. math::
            p(x) = x

        else.

        Examples
        --------
        Define the nadir and the points which should dominate

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
                 epsilon: Union[float, np.ndarray]):
        """Specify the nadir and the to be dominated point

        Parameters
        ----------
        nadir : np.ndarray
            nadir (dominated by all points) stored in 1 dimensional array of length n

        epsilon_avoiding_points : np.ndarray
            avoided points stored in 2-dimensional array with first dimension corresponding to the points

        epsilon : Union[float, np.ndarray])
            value which is subtracted from avoided points
        """

        if nadir.shape != epsilon_avoiding_points.shape and nadir.shape != epsilon_avoiding_points[0].shape:
            raise ValueError('Nadir and avoiding points need to be 2-dimensional numpy arrays of equal shape!')

        # TODO: error handling rest

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
        # TODO: add dimensionality check
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
