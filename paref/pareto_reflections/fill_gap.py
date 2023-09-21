import numpy as np
from paref.pareto_reflections.minimize_weighted_norm_to_utopia import MinimizeWeightedNormToUtopia
import scipy as sp


class FillGap(MinimizeWeightedNormToUtopia):
    """Fill the gap spanned by m points of dimension m


    When to use
    -----------
    This Pareto reflection should be used if a Pareto point is desired which closes the gap spanned by
    specified points

    .. note::

        This function works best if the points are Pareto optimal.


    What it does
    ------------
    The Pareto points of this map are (approximately) the Pareto points which are closest to the center of the gap.

    Mathematical formula
    --------------------

    .. math::
        TBA

    Examples
    --------
    TBA
    """

    def __init__(self,
                 dimension_domain: int,
                 gap_points: np.ndarray,
                 epsilon: float = 0.01):
        """Specify the gap and some utopia point

        Parameters
        ----------
        dimension_domain : np.ndarray
            dimension of the domain of the Pareto reflection

        gap_points : np.ndarray
            m points of dimension m defining the gap with first dimension corresponding to the number of points

        epsilon : float default 0.01
            epsilon of underlying weighted norm

        """
        if len(gap_points) != dimension_domain:
            raise ValueError('The number of gap defining points must be equal to the dimension of the domain.')
        self.epsilon = epsilon
        self.gap_points = gap_points
        self.center = np.sum(gap_points, axis=0) / len(gap_points)

        # calculate orthogonal basis
        span = gap_points - self.center
        self.orthogonal_basis = sp.linalg.orth(span.T).T
        self.g = lambda x: np.linalg.norm(np.sum(np.array([np.dot(x - self.center, basis_vector) * basis_vector
                                                           for basis_vector in self.orthogonal_basis]), axis=0))

        self._dimension_domain = dimension_domain

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x) * self.epsilon + self.g(x)

    @property
    def dimension_codomain(self) -> int:
        return 1

    @property
    def dimension_domain(self) -> int:
        return self._dimension_domain
