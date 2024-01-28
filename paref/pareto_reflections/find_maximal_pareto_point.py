import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection


class FindMaximalParetoPoint(ParetoReflection):
    """Find a Pareto point which represents a trade-off in all components

    When to use
    -----------
    This Pareto reflection should be used if a Pareto point is desired which represents a trade-off in all components.

    What it does
    ------------
    The Pareto points of this map are the ones which are closest to the theoretical global optimum after normalization.

    .. warning::

            This Pareto reflection assumes that the minima of components was already (approximately) found.

    Examples
    --------
    # TBA: add
    """

    def __init__(self,
                 blackbox_function: BlackboxFunction,
                 potency: int = 4,):
        """

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            blackbox function to which this reflection is applied

        """
        self._dimension_domain = blackbox_function.dimension_target_space
        self.potency = potency
        self.normalize = lambda x: x-np.min(blackbox_function.y, axis=0)
        # normalize such that components are (approximately) in the range [0,1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.normalize(x), ord=self.potency)

    @property
    def dimension_codomain(self) -> int:
        return 1

    @property
    def dimension_domain(self) -> int:
        return self._dimension_domain
