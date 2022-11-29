import numpy as np

from paref.pareto_reflecting_library.functions.interfaces.pareto_reflecting_function import ParetoReflectingFunction


class Composing(ParetoReflectingFunction):
    """
    Compose Pareto reflecting functions.
    Maps x to pareto_reflecting_function_2(pareto_reflecting_function_1(x)).
    """
    def __init__(self,
                 pareto_reflecting_function_1: ParetoReflectingFunction,
                 pareto_reflecting_function_2: ParetoReflectingFunction):
        self.pareto_reflecting_function_1 = pareto_reflecting_function_1
        self.pareto_reflecting_function_2 = pareto_reflecting_function_2

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.pareto_reflecting_function_2(self.pareto_reflecting_function_1(x))
