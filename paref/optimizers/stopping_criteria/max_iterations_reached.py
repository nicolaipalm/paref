from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.optimizers.stopping_criteria import StoppingCriteria


class MaxIterationsReached(StoppingCriteria):
    def __init__(self, max_iterations: int = 50):
        self._iteration_step = 0
        self._max_iterations = max_iterations

    def __call__(self):
        if self._iteration_step < self._max_iterations:
            self._iteration_step += 1
            return False

        else:
            return True
