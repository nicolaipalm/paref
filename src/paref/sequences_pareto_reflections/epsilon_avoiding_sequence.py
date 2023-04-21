import numpy as np

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.optimizers.helper_functions.return_pareto_front import return_pareto_front
from paref.pareto_reflections.epsilon_avoiding import EpsilonAvoiding
from paref.interfaces.pareto_reflections.pareto_reflecting_function import ParetoReflectingFunction
from paref.pareto_reflections.operations.composing import Composing
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflecting_functions import \
    SequenceParetoReflectingFunctions


class EpsilonAvoidingSequence(SequenceParetoReflectingFunctions):
    def __init__(self, nadir: np.ndarray, pareto_reflecting_function: ParetoReflectingFunction, epsilon: float = 0):
        self._nadir = nadir
        self._epsilon = epsilon
        self._pareto_reflecting_function = pareto_reflecting_function

    def next(self, blackbox_function: BlackboxFunction) -> ParetoReflectingFunction:
        return Composing(
            EpsilonAvoiding(nadir=self._nadir,
                            epsilon=self._epsilon,
                            epsilon_avoiding_points=return_pareto_front(blackbox_function.y)),
            self._pareto_reflecting_function)