# iteratively defined -> yield statement
from abc import abstractmethod

from paref.function_library.interfaces.function import Function


class MOOExpress:
    @abstractmethod
    def __call__(self,
                 blackbox_function: Function,):
        pass
