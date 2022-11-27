from abc import abstractmethod

import numpy as np


class ParetoReflectingFunction:
    """
    Interface fot Pareto reflecting function.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass
