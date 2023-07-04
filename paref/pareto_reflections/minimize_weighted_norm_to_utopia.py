from typing import Union

import numpy as np

from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection


class MinimizeWeightedNormToUtopia(ParetoReflection):
    """Find the Pareto point closest to some utopia point
    # TBA: not a norm in general rather a polynomial

    .. note::

        # TBA: add
        This Pareto reflection is highly flexible. TBA (used for finding lots of properties, for application: if
        utopia point represents only point which we are looking for...)

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

    def __init__(self, utopia_point: np.ndarray, potency: Union[np.ndarray, float], scalar: np.ndarray):
        """Specify the utopia point, the potency and the scalar used in the weighted p-norm

        Parameters
        ----------
        utopia_point :
        np.ndarray
            utopia point stored in 1 dimensional array of length n
        potency :
        np.ndarray
            potency (i.e. value of p for the used p-norm)
            # TBA: not p norm but p norm power p
        scalar :
        np.ndarray
            scalar vector stored in 1 dimensional array of length n
        """
        self.potency = potency
        self.scalar = scalar
        self.utopia_point = utopia_point

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) != 1:
            raise ValueError(f'Input x must be of dimension 1! Shape of x is {x.shape}.')
        return np.sum((self.scalar*(x-self.utopia_point))**self.potency)

    @property
    def dimension_codomain(self) -> int:
        return 1

    @property
    def dimension_domain(self) -> int:
        return len(self.utopia_point)
