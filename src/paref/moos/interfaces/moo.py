# iteratively defined -> yield statement
from abc import abstractmethod

from paref.function_library.interfaces.function import Function
from paref.pareto_reflecting_library.sequences.interfaces.sequence_pareto_reflecting_functions import \
    SequenceParetoReflectingFunctions
from paref.stopping_criteria.interfaces.stopping_criteria import StoppingCriteria


class MOO:
    @abstractmethod
    def __call__(self,
                 blackbox_function: Function,
                 pareto_reflecting_sequence: SequenceParetoReflectingFunctions,
                 stopping_criteria: StoppingCriteria):
        pass
