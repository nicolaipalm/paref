# iteratively defined -> yield statement
from abc import abstractmethod

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction


class MOOAlgorithm:
    @abstractmethod
    def __call__(self,
                 blackbox_function: BlackboxFunction, ):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
