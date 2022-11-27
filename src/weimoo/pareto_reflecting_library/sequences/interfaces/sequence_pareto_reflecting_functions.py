from abc import abstractmethod

from weimoo.function_library.interfaces.function import Function
from weimoo.pareto_reflecting_library.functions.interfaces.pareto_reflecting_function import ParetoReflectingFunction


class SequenceParetoReflectingFunctions:

    @abstractmethod
    def next(self, blackbox_function: Function) -> ParetoReflectingFunction:
        pass
