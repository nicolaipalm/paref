from typing import List

from weimoo.function_library.interfaces.function import Function
from weimoo.pareto_reflecting_library.functions.interfaces.pareto_reflecting_function import ParetoReflectingFunction
from weimoo.pareto_reflecting_library.sequences.interfaces.sequence_pareto_reflecting_functions import \
    SequenceParetoReflectingFunctions
from weimoo.stopping_criteria.interfaces.stopping_criteria import StoppingCriteria


class RepeatingSequence(SequenceParetoReflectingFunctions):
    def __init__(self, stopping_criteria: StoppingCriteria,
                 pareto_reflecting_functions: List[ParetoReflectingFunction]):
        self._stopping_criteria = stopping_criteria
        self._pareto_reflecting_functions = pareto_reflecting_functions
        self._iter = 0

    def next(self, blackbox_function: Function) -> ParetoReflectingFunction:
        if self._stopping_criteria(blackbox_function) and len(self._pareto_reflecting_functions) - 1 > self._iter:
            self._iter += 1

        return self._pareto_reflecting_functions[self._iter]
