from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.moo_algorithms.stopping_criteria import StoppingCriteria


class TimeLimitReached(StoppingCriteria):
    def __call__(self, blackbox_function: BlackboxFunction) -> bool:
        raise NotImplementedError
