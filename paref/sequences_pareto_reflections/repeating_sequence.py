from typing import List

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflecting_function import ParetoReflectingFunction
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflecting_functions import \
    SequenceParetoReflectingFunctions
from paref.interfaces.optimizers.stopping_criteria import StoppingCriteria


class RepeatingSequence(SequenceParetoReflectingFunctions):
    def __init__(self, stopping_criteria: StoppingCriteria,
                 pareto_reflecting_functions: List[ParetoReflectingFunction]):
        self._stopping_criteria = stopping_criteria
        self._pareto_reflecting_functions = pareto_reflecting_functions
        self._iter = 0

    def next(self, blackbox_function: BlackboxFunction) -> ParetoReflectingFunction:
        if self._stopping_criteria(blackbox_function) and len(self._pareto_reflecting_functions) - 1 > self._iter:
            self._iter += 1

        return self._pareto_reflecting_functions[self._iter]
