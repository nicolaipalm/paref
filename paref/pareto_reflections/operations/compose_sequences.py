from typing import Optional, Union

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.pareto_reflections.operations.compose_reflections import ComposeReflections


# todo: as method of pareto_reflections (use function..)

class ComposeSequences(SequenceParetoReflections):
    """Compose two Pareto reflections and obtain a new Pareto reflection

    """

    def __init__(self,
                 sequence_1: Union[SequenceParetoReflections, ParetoReflection],
                 sequence_2: Union[SequenceParetoReflections, ParetoReflection]):
        """

        Parameters
        ----------
        pareto_reflecting_function_1 : ParetoReflection
            Pareto reflection which is applied first

        pareto_reflecting_function_2 : ParetoReflection
            Pareto reflection which is applied second
        """
        self.sequence_1 = sequence_1
        self.sequence_2 = sequence_2

    def next(self, blackbox_function: BlackboxFunction) -> Optional[ParetoReflection]:
        if isinstance(self.sequence_1, ParetoReflection):
            reflection_1 = self.sequence_1

        elif isinstance(self.sequence_1, SequenceParetoReflections):
            reflection_1 = self.sequence_1.next(blackbox_function)

        else:
            raise ValueError('sequence_1 must be an instance of SequenceParetoReflections or ParetoReflection!')

        if reflection_1 is None:
            return None

        if isinstance(self.sequence_2, ParetoReflection):
            reflection_2 = self.sequence_2

        elif isinstance(self.sequence_2, SequenceParetoReflections):
            reflection_2 = self.sequence_2.next(blackbox_function)

        else:
            raise ValueError('sequence_2 must be an instance of SequenceParetoReflections or ParetoReflection!')

        if reflection_2 is None:
            return None

        if reflection_1.dimension_codomain != reflection_2.dimension_domain:
            raise ValueError('Codomain of reflection_1 and domain of reflection_2 must match! '
                             f'Codomain of reflection_1 is {reflection_1.dimension_codomain} '
                             f'and domain of reflection_2 is {reflection_2.dimension_domain}.')

        return ComposeReflections(reflection_1, reflection_2)
