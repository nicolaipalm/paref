import numpy as np

from paref.function_library.interfaces.function import Function
from paref.pareto_reflecting_library.functions.epsilon_avoiding import EpsilonAvoiding
from paref.pareto_reflecting_library.functions.interfaces.pareto_reflecting_function import ParetoReflectingFunction
from paref.pareto_reflecting_library.functions.operations.composing import Composing
from paref.pareto_reflecting_library.sequences.interfaces.sequence_pareto_reflecting_functions import \
    SequenceParetoReflectingFunctions


class EpsilonAvoidingSequence(SequenceParetoReflectingFunctions):
    def __init__(self, nadir: np.ndarray, pareto_reflecting_function: ParetoReflectingFunction, epsilon: float = 0):
        self._nadir = nadir
        self._epsilon = epsilon
        self._pareto_reflecting_function = pareto_reflecting_function

    def next(self, blackbox_function: Function) -> ParetoReflectingFunction:
        return Composing(
            EpsilonAvoiding(nadir=self._nadir,
                            epsilon=self._epsilon,
                            epsilon_avoiding_points=blackbox_function.y),
            self._pareto_reflecting_function)
