import numpy as np

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflecting_function import ParetoReflectingFunction
from paref.pareto_reflections.operations.composing import Composing
from paref.pareto_reflections.restricting import Restricting
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflecting_functions import \
    SequenceParetoReflectingFunctions


class RestrictingSequence(SequenceParetoReflectingFunctions):
    def __init__(self,
                 nadir: np.ndarray,
                 pareto_reflecting_function: ParetoReflectingFunction,
                 restricting_point: np.ndarray, ):
        self._nadir = nadir
        self._restricting_point = restricting_point
        self._pareto_reflecting_function = pareto_reflecting_function

    def next(self,
             blackbox_function: BlackboxFunction) -> ParetoReflectingFunction:
        print("Restricting point:", self._restricting_point)
        return Composing(
            Restricting(nadir=self._nadir,
                        restricting_point=self._restricting_point),
            self._pareto_reflecting_function)
