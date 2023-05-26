from abc import abstractmethod

import numpy as np


class ParetoReflectingFunction:
    """Interface for Pareto reflections

    Documentation should contain:

    When to use
    -----------
    This sequence should be used if...

    What it does
    ------------
    The sequence...

    Examples
    --------

    TODO: into contribution
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass
