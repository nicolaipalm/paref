from abc import abstractmethod

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction


class StoppingCriteria:
    @abstractmethod
    def __call__(self, blackbox_function: BlackboxFunction) -> bool:
        pass



