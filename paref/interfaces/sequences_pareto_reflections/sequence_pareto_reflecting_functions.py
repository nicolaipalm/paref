from abc import abstractmethod

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflecting_function import ParetoReflectingFunction


class SequenceParetoReflectingFunctions:

    @abstractmethod
    def next(self, blackbox_function: BlackboxFunction) -> ParetoReflectingFunction:
        pass
