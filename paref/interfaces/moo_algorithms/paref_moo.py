from abc import abstractmethod
from typing import Optional, List, Union

import numpy as np

from paref.black_box_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import \
    SequenceParetoReflections
from paref.interfaces.moo_algorithms.stopping_criteria import StoppingCriteria
from paref.pareto_reflections.operations.compose_reflections import ComposeReflections
from paref.pareto_reflections.operations.compose_sequences import ComposeSequences


class ParefMOO:
    @abstractmethod
    def apply_moo_operation(self,
                            blackbox_function: BlackboxFunction,
                            ):
        raise NotImplementedError

    # TODO: how to supported domain? Or needed?
    @property
    @abstractmethod
    def supported_codomain_dimensions(self) -> List[int]:
        # If None then all codomain dimensions are supported
        raise NotImplementedError

    def __call__(self,
                 blackbox_function: BlackboxFunction,
                 stopping_criteria: StoppingCriteria,
                 ):

        sequence_of_pareto_reflections = self.sequence_of_pareto_reflections

        if sequence_of_pareto_reflections is None:
            while not stopping_criteria(blackbox_function):
                # TODO: monitoring
                self.apply_moo_operation(blackbox_function)

        else:
            self.apply_to_sequence(blackbox_function, sequence_of_pareto_reflections, stopping_criteria)

    def apply_to_sequence(self,
                          blackbox_function: BlackboxFunction,
                          sequence_pareto_reflections: Union[SequenceParetoReflections, ParetoReflection],
                          stopping_criteria: StoppingCriteria,
                          ):
        moo_sequence_of_pareto_reflections = self.sequence_of_pareto_reflections
        if moo_sequence_of_pareto_reflections is not None:
            sequence_pareto_reflections = ComposeSequences(sequence_pareto_reflections,
                                                           moo_sequence_of_pareto_reflections)

        while not stopping_criteria(blackbox_function):
            if isinstance(sequence_pareto_reflections, SequenceParetoReflections):
                # compose: caution what if one returns None
                pareto_reflection = sequence_pareto_reflections.next(blackbox_function)
                if pareto_reflection is None:
                    print('End of sequence reached. Algorithm stopped.')
                    break
                else:
                    composition_function = CompositionWithParetoReflection(blackbox_function=blackbox_function,
                                                                           pareto_reflection=pareto_reflection)

            elif isinstance(sequence_pareto_reflections, ParetoReflection):
                composition_function = CompositionWithParetoReflection(blackbox_function=blackbox_function,
                                                                       pareto_reflection=sequence_pareto_reflections)

            else:
                raise ValueError(
                    'sequence_pareto_reflections must be an instance of sequence '
                    'of Pareto reflections or a single Pareto reflection!')
            self.apply_moo_operation(composition_function)

    @property
    def sequence_of_pareto_reflections(self) -> Union[SequenceParetoReflections, ParetoReflection, None]:
        return None


class CompositionWithParetoReflection(BlackboxFunction):
    def __init__(self, blackbox_function: BlackboxFunction, pareto_reflection: ParetoReflection):
        if blackbox_function.dimension_target_space != pareto_reflection.dimension_domain:
            raise ValueError(
                f'Dimension of target space ({blackbox_function.dimension_target_space}) of blackbox function and '
                f'domain ({pareto_reflection.dimension_domain}) of Pareto reflection must match!')
        self._blackbox_function = blackbox_function
        self._pareto_reflection = pareto_reflection
        self._evaluations = blackbox_function._evaluations

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._pareto_reflection(self._blackbox_function(x))

    @property
    def dimension_design_space(self) -> int:
        return self._blackbox_function.dimension_design_space

    @property
    @abstractmethod
    def dimension_target_space(self) -> int:
        return self._pareto_reflection.dimension_codomain

    @property
    @abstractmethod
    def design_space(self) -> Union[Bounds]:
        return self._blackbox_function.design_space
