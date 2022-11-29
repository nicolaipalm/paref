from abc import abstractmethod

from paref.function_library.interfaces.function import Function
from paref.pareto_reflecting_library.functions.interfaces.pareto_reflecting_function import ParetoReflectingFunction


class SequenceParetoReflectingFunctions:

    @abstractmethod
    def next(self, blackbox_function: Function) -> ParetoReflectingFunction:
        pass
