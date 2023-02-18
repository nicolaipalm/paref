import numpy as np

from paref.function_library.interfaces.function import Function
from paref.pareto_reflecting_library.functions.interfaces.pareto_reflecting_function import ParetoReflectingFunction
from paref.pareto_reflecting_library.functions.operations.composing import Composing
from paref.pareto_reflecting_library.functions.restricting import Restricting
from paref.pareto_reflecting_library.sequences.interfaces.sequence_pareto_reflecting_functions import \
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
        print("Restricting point:", self._restricting_point)
        return Composing(
            Restricting(nadir=self._nadir,
                        restricting_point=self._restricting_point),
            self._pareto_reflecting_function)
