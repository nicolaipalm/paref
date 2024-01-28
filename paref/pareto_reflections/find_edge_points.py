import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.pareto_reflections.minimize_g import MinGParetoReflection


class FindEdgePoints(MinGParetoReflection):
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
                 dimension: int,
                 blackbox_function: BlackboxFunction, ):
        self.bbf = blackbox_function
        self._counter = 0
        self.g = lambda x: np.sum(x)-x[dimension]
