from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.pareto_reflections.minimize_g import MinGParetoReflection


class Find1ParetoPoints(MinGParetoReflection):
    """Find a Pareto point which is minimal in some specified component

    When to use
    -----------
    This Pareto reflection should be used if a Pareto point is desired which lie on the boundary of the Pareto front,
    i.e. are minimal in one component.

    .. note::

        In to dimensions, the edge points of the Pareto front are given by the boundary of the Pareto front.
        I.p. in two dimensions, this Pareto reflection searches for the edge points of the Pareto front.

    What it does
    ------------
    The Pareto points of this map are the ones which minimize the weighted sum where one component
    is given much more weight than the others.

    Mathematical formula
    --------------------

    .. math::
        p(x) = \sum_{i=1,...,n,i\\neq j}\\epsilon x_{i}+ x_j

    where :math:`j` is the component in which the minimum is searched.
    """

    def __init__(self,
                 dimension: int,
                 blackbox_function: BlackboxFunction, ):
        """

        Parameters
        ----------
        dimension :
            component in which the minimum is searched

        blackbox_function :
            blackbox function to which this reflection is applied
        """
        self.dimension = dimension
        super().__init__(blackbox_function=blackbox_function)

    @property
    def g(self):
        return lambda x: x[self.dimension]
