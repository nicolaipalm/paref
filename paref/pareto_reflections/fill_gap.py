from typing import Callable

import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.pareto_reflections.minimize_g import MinGParetoReflection
import scipy as sp


class FillGap(MinGParetoReflection):
    """Fill the gap spanned by m points of dimension m


    When to use
    -----------
    This Pareto reflection should be used if a Pareto point is desired which closes the gap spanned by
    specified points

    .. note::

        This function works best if those points are Pareto optimal.


    What it does
    ------------
    The Pareto points of this map are (approximately) the Pareto points which are closest to the center of the gap.

    """

    def __init__(self,
                 blackbox_function: BlackboxFunction,
                 gap_points: np.ndarray, ):
        """Specify the gap and some utopia point

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            blackbox function to which this reflection is applied

        gap_points : np.ndarray
            m points of dimension m defining the gap with first dimension of the numpy array
            corresponding to the number of points
        """
        super().__init__(blackbox_function=blackbox_function)
        dimension_domain = blackbox_function.dimension_target_space
        if len(gap_points) != dimension_domain:
            raise ValueError('The number of gap defining points must be equal to the dimension of the domain.')
        self.gap_points = gap_points
        self.center = np.sum(gap_points, axis=0) / len(gap_points)

        # calculate orthogonal basis
        span = gap_points - self.center
        self.orthogonal_basis = sp.linalg.orth(span.T).T

        self._g = lambda x: np.linalg.norm(np.sum(np.array([np.dot(x - self.center, basis_vector) * basis_vector
                                                            for basis_vector in self.orthogonal_basis]), axis=0))

    @property
    def g(self, ) -> Callable:
        return self._g
