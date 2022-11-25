import numpy as np


class ParetoReflectingFunction:
    """
    The weight function transforms multiple dimensional input into scalars.
    It satisfies that a minimum of the weight function is a pareto optimum (but n.n. vice versa).
    """

    def __call__(self, x: np.ndarray) -> float:
        pass
