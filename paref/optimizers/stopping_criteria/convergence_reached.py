import numpy as np

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.sequences_pareto_reflections.stopping_criteria import StoppingCriteria


class ConvergenceReached(StoppingCriteria):
    def __init__(self,
                 blackbox_function: BlackboxFunction,
                 epsilon: float = 1e-3,
                 ):
        self._epsilon = epsilon
        self._blackbox_function = blackbox_function

    def __call__(self):
        norm = np.linalg.norm(self._blackbox_function.evaluations[-1][1] -
                              self._blackbox_function.evaluations[-2][1])

        if norm > self._epsilon:
            return False

        else:
            print(f'Convergence reached. The l2-distance of the last two points found is {norm}.')
            return True
