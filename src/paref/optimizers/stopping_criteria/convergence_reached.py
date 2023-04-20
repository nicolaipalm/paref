import numpy as np

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.optimizers.stopping_criteria import StoppingCriteria


class ConvergenceReached(StoppingCriteria):
    def __init__(self, epsilon: float = 1e-3):
        self._epsilon = epsilon

    def __call__(self, blackbox_function: BlackboxFunction):
        norm = np.linalg.norm(blackbox_function.evaluations[-1][1] -
                              blackbox_function.evaluations[-2][1])

        if norm > self._epsilon:
            return False

        else:
            print(f"Convergence reached. The l2-distance of the last two points found is {norm}.")
            return True
