import numpy as np

from weimoo.function_library.interfaces.function import Function
from weimoo.pareto_reflecting_library.functions.interfaces.pareto_reflecting_function import ParetoReflectingFunction
from weimoo.pareto_reflecting_library.functions.operations.composing import Composing
from weimoo.pareto_reflecting_library.functions.restricting import Restricting
from weimoo.pareto_reflecting_library.sequences.interfaces.sequence_pareto_reflecting_functions import \
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
             blackbox_function: Function) -> ParetoReflectingFunction:
        return Composing(
            Restricting(nadir=self._nadir,
                        restricting_point=self._restricting_point),
            self._pareto_reflecting_function)
