import warnings

import numpy as np
from paref.pareto_reflections.minimize_weighted_norm_to_utopia import MinimizeWeightedNormToUtopia


class FillGap2D(MinimizeWeightedNormToUtopia):
    """Weighted norm to some utopia point (Pareto Reflection)

    When to use
    -----------
    This Pareto reflection should be used if a Pareto point is desired which is closest to some utopia point.

    What it does
    ------------
    The Pareto points of this map are the ones which minimize the (weighted) distance to the utopia point.

    Mathematical formula
    --------------------

    .. math::
        p(x) = \sum_{i=1,...,n}(a_{i}x_{i}-a_{i}u_{i})^p

    where n denotes the input dimension, u denotes some utopia point, a denotes some vector of dimension n
    with strictly positive entries and p is the potency.

    u,a and p are fixed when the WeightedNormToUtopia is initialized.
    Calling returns the weighted p-norm to the given utopia point.

    Examples
    --------
    Define the utopia point, potency (p) and the scalar

    >>> import numpy as np
    >>> utopia_point, potency, scalar = np.zeros(2), np.array([2]), np.ones(2)

    Initialize the WeightedNormToUtopia

    >>> pareto_reflection = MinimizeWeightedNormToUtopia(utopia_point=utopia_point, potency=potency, scalar=scalar)

    Calling it to (1,1), i.e. calculating

    .. math::
        p(1,1)=(1-0)^2+(1-0)^2=2

    yields:

    >>> pareto_reflection(np.ones(2))
    2
    """

    def __init__(self,
                 point_1: np.ndarray,
                 point_2: np.ndarray,
                 utopia_point: np.ndarray,
                 potency: int = 6,
                 ):
        """Specify the utopia point, the potency and the scalar used in the weighted p-norm

        Parameters
        ----------
        utopia_point :
        np.ndarray
            utopia point stored in 1 dimensional array of length n
        potency :
        np.ndarray
            potency (i.e. value of p for the used p-norm)
        scalar :
        np.ndarray
            scalar vector stored in 1 dimensional array of length n
        """

        dimension_domain = 2
        m = 1 / 2 * (point_1 + point_2)  # middle point
        y = point_1 - point_2  # helper
        v = np.array([1, -(y[0] / y[1])])  # normal vector
        print('Normal vector: ',v)
        lamb = (utopia_point[0] - m[0]) / v[0]  # parameter of utopia point projected to normal
        if (m + lamb * v)[1] > utopia_point[1]:
            lamb = (utopia_point[1] - m[1]) / v[1]

        utopia_point = m + lamb * v  # utopia point projected to normal st previous utopia point is dominated

        self.potency = potency * np.ones(dimension_domain)
        self.scalar = (1/v) ** potency
        self.utopia_point = utopia_point
        print('Utopia projection: ', self.utopia_point)

    @property
    def dimension_codomain(self) -> int:
        return 1

    @property
    def dimension_domain(self) -> int:
        return 2
