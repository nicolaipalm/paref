from abc import abstractmethod

from paref.function_library.interfaces.function import Function


class StoppingCriteria:
    @abstractmethod
    def __call__(self, blackbox_function: Function) -> bool:
        pass



