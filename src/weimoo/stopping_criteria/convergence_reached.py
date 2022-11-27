import numpy as np

from weimoo.function_library.interfaces.function import Function
from weimoo.stopping_criteria.interfaces.stopping_criteria import StoppingCriteria


class ConvergenceReached(StoppingCriteria):
    def __init__(self, epsilon: float = 1e-3):
        self._epsilon = epsilon

    def __call__(self, blackbox_function: Function):
        norm = np.linalg.norm(blackbox_function.evaluations[-1][1] -
                              blackbox_function.evaluations[-2][1])

        if norm > self._epsilon:
            return False

        else:
            print(f"Convergence reached. The norm of the last two found points is {norm}.")
            return True
