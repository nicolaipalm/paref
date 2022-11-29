from paref.function_library.interfaces.function import Function
from paref.stopping_criteria.interfaces.stopping_criteria import StoppingCriteria


class MaxIterationsReached(StoppingCriteria):
    def __init__(self, max_iterations: int = 50):
        self._iteration_step = 0
        self._max_iterations = max_iterations

    def __call__(self, blackbox_function: Function):
        if self._iteration_step < self._max_iterations:
            self._iteration_step += 1
            return False

        else:
            return True
