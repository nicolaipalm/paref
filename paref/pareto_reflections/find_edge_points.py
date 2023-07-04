import numpy as np

from paref.pareto_reflections.minimize_weighted_norm_to_utopia import MinimizeWeightedNormToUtopia


class FindEdgePoints(MinimizeWeightedNormToUtopia):
    """Find the edge points of the Pareto front

    .. warning::

        This Pareto reflection assumes that there exist edge points

    When to use
    -----------
    This Pareto reflection should be used if the edge points of the Pareto front are searched.

    .. note::

        In to dimensions, the edge points of the Pareto front always exist.

    What it does
    ------------
    The Pareto points of this map are the ones which minimize the weighted sum where one component
    is given much smaller weight than the others.

    Mathematical formula
    --------------------

    .. math::
        p(x) = \sum_{i=1,...,n,i\\neq j}\\epsilon x_{i}+ x_j

    where :math:`j` is the component in which the minimum is searched.

    Examples
    --------
    # TBA: add
    """

    def __init__(self, dimension_domain: int, dimension: int, epsilon: float = 1e-3):
        """Specify the dimension of the input domain and the component of which the edge point is searched

        .. warning::

            The smaller epsilon, the better. However, picking an epsilon too small may lead to an
            unstable optimization.

        Parameters
        ----------
        dimension_domain : int
            dimension of domain (i.e. dimension of target space of blackbox function)

        dimension : int
            component of which the edge point is searched

        epsilon : float default 1e-3
            weight on the component
        """
        self.epsilon = epsilon
        self.dimension = dimension
        self._dimension_domain = dimension_domain
        self.potency = np.ones(dimension_domain)
        scalar = np.ones(dimension_domain)
        scalar[self.dimension] = epsilon
        self.scalar = scalar
        self.utopia_point = np.zeros(dimension_domain)

    @property
    def dimension_codomain(self) -> int:
        return 1

    @property
    def dimension_domain(self) -> int:
        return self._dimension_domain
