import numpy as np
from paref.pareto_reflections.minimize_weighted_norm_to_utopia import MinimizeWeightedNormToUtopia


class FillGap2D(MinimizeWeightedNormToUtopia):
    """Fill the gap between two to be specified points in two dimensions

    .. warning::

        Although this Pareto reflection theoretically closes every gap, if the utopia point is far away from
        the gap, algorithms based on this Pareto reflection tend to be numerically unstable.
        It is recommended to choose an utopia point which is as close to the gap as possible.

    When to use
    -----------
    This Pareto reflection should be used if a Pareto point is desired which lies between two specified points

    .. note::

        This function works best if the two points are Pareto optimal.


    What it does
    ------------
    The Pareto points of this map are the ones which sit in the middle of the two given points (provided
    that such a point exists).

    Mathematical formula
    --------------------

    .. math::
        TBA

    Examples
    --------
    TBA: Add
    """

    def __init__(self, point_1: np.ndarray,
                 point_2: np.ndarray,
                 utopia_point: np.ndarray,
                 potency: int = 6):
        """Specify the gap and some utopia point

        Parameters
        ----------
        point_1 : np.ndarray
            first point defining the gap

        point_2 : np.ndarray
            second point defining the gap

        utopia_point : np.ndarray
            utopia point

        potency : int default 6
            potency of underlying weighted norm
        """
        if point_1.shape != (self.dimension_domain,) or point_2.shape != (
                self.dimension_domain,) or utopia_point.shape != (self.dimension_domain,):
            raise ValueError('Both points and utopia points must be 1 dimensional arrays of length 2! Shape of '
                             f'utopia_point: {utopia_point.shape}'
                             f'\n point_1: {point_1.shape} '
                             f'\n point_1: {point_2.shape}')

        dimension_domain = self.dimension_domain
        m = 1 / 2 * (point_1 + point_2)  # middle point
        y = point_1 - point_2  # helper
        v = np.array([1, -(y[0] / y[1])])  # normal vector
        lamb = (utopia_point[0] - m[0]) / v[0]  # parameter of utopia point projected to normal
        if (utopia_point[1] - m[1]) / v[1] > lamb:
            lamb = (utopia_point[1] - m[1]) / v[1]

        self.potency = potency * np.ones(dimension_domain)
        self.scalar = (1 / v)
        self.utopia_point = m + lamb * v  # utopia point projected to normal st previous utopia point is dominated

    @property
    def dimension_codomain(self) -> int:
        return 1

    @property
    def dimension_domain(self) -> int:
        return 2
