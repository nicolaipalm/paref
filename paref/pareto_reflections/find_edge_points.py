import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection


class FindEdgePoints(ParetoReflection):
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

    def __init__(self,
                 blackbox_function: BlackboxFunction,
                 dimension: int,
                 epsilon: float = 1e-3):
        """Specify the dimension of the input domain and the component of which the edge point is searched

        .. warning::

            The smaller epsilon, the better. However, picking an epsilon too small may lead to an
            unstable optimization.

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            blackbox function to which this reflection is applied

        dimension : int
            component of which the edge point is searched

        epsilon : float default 1e-3
            weight on the component
        """
        self.epsilon = epsilon
        self.dimension = dimension
        self._dimension_domain = blackbox_function.dimension_target_space
        self.potency = np.ones(self._dimension_domain)
        scalar = np.ones(self._dimension_domain)
        scalar[self.dimension] = epsilon
        self.scalar = scalar
        self.utopia_point = np.zeros(self._dimension_domain)

        if len(blackbox_function.y) == 0:
            self._normalization_factor = 1
        else:
            self._normalization_factor = (np.abs(np.min(blackbox_function.y, axis=0)) + 1)
        # TODO: normalzation might be a problem in the future
        # normalize such that components are (approximately) in the range [0,1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sum(self.scalar * x / self._normalization_factor)

    @property
    def dimension_codomain(self) -> int:
        return 1

    @property
    def dimension_domain(self) -> int:
        return self._dimension_domain
