# iteratively defined -> yield statement
from abc import abstractmethod

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflecting_functions import \
    SequenceParetoReflectingFunctions


class MOO:
    @abstractmethod
    def __call__(self,
                 blackbox_function: BlackboxFunction,
                 pareto_reflecting_sequence: SequenceParetoReflectingFunctions,
                 ):
        # if pareto_reflecting_sequence returns None then optimizer should stop
        pass
