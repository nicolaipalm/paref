from abc import abstractmethod

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.optimizers.stopping_criteria import StoppingCriteria


class LogicalOrStoppingCriteria(StoppingCriteria):
    def __init__(self,
                 stopping_criteria_1: StoppingCriteria,
                 stopping_criteria_2: StoppingCriteria):
        self.stopping_criteria_1 = stopping_criteria_1
        self.stopping_criteria_2 = stopping_criteria_2

    @abstractmethod
    def __call__(self, blackbox_function: BlackboxFunction) -> bool:
        return self.stopping_criteria_1(blackbox_function=blackbox_function) or self.stopping_criteria_2(
            blackbox_function=blackbox_function)
