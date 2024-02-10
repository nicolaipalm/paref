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
            Use the ``Find1ParetoPoints`` Pareto reflection to find the minima of components.


    """

    def __init__(self,
                 blackbox_function: BlackboxFunction,
                 potency: int = 4, ):
        """

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            blackbox function to which this reflection is applied

        potency : int
            p rank of the underlying p-norm

        """
        self._dimension_domain = blackbox_function.dimension_target_space
        self.potency = potency
        self._counter = 0
        self.bbf = blackbox_function

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._counter < 2:
            self._counter += 1
            if len(self.bbf.y) == 0:
                self.m = 0
            else:
                self.m = np.min(self.bbf.y, axis=0)
        return np.linalg.norm(x - self.m, ord=self.potency)

    @property
    def dimension_codomain(self) -> int:
        return 1

    @property
    def dimension_domain(self) -> int:
        return self._dimension_domain
