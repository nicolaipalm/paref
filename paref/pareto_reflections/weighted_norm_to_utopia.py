import numpy as np

from paref.interfaces.pareto_reflections.pareto_reflecting_function import ParetoReflectingFunction


class WeightedNormToUtopia(ParetoReflectingFunction):
    """Weighted norm to some utopia point (Pareto Reflection)

    When to use
    -----------
    This Pareto reflection should be used if a Pareto point is desired which is closest to some utopia point.

    What it does
    ------------
    The Pareto points of this map are the ones which minimize the (weighteed) distance to the utopia point.

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
    >>> utopia_point, potency, scalar = np.zeros(2),2,np.ones(2)

    Initialize the WeightedNormToUtopia

    >>> pareto_reflection = WeightedNormToUtopia(utopia_point=utopia_point, potency=potency, scalar=scalar)

    Calling it to (1,1), i.e. calculating

    .. math::
        p(1,1)=(1-0)^2+(1-0)^2=2

    yields:

    >>> pareto_reflection(np.ones(2))
    2
    """

    def __init__(self, utopia_point: np.ndarray, potency: int, scalar: np.ndarray):
        """Specify the utopia point, the potency and the scalar used in the weighted p-norm

        Parameters
        ----------
        utopia_point :
        np.ndarray
            utopia point stored in 1 dimensional array of length n
        potency :
        int
            potency (i.e. value of p for the used p-norm)
        scalar :
        np.ndarray
            scalar vector stored in 1 dimensional array of length n
        """
        self.potency = potency
        self.scalar = scalar
        self.utopia_point = utopia_point

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Calculate the weighted norm (p-norm) to some utopia point

        Parameters
        ----------
        x :
        np.ndarray
            input vector stored in 1 dimensional array

        Returns
        -------
        np.ndarray
            value of the weighted p-norm to some utopia point, i.e. 1 dimensional array of length 1

        """
        return np.sum(self.scalar * ((x - self.utopia_point) ** self.potency))
