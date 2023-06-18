from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer
from paref.pareto_reflection_sequences.two_dimensional.fill_gaps_of_pareto_front_sequence_2d import \
    FillGapsOfParetoFrontSequence2D


class FillGapsOfParetoFront2D(GPRMinimizer):
    @property
    def sequence_of_pareto_reflections(self) -> SequenceParetoReflections:
        return FillGapsOfParetoFrontSequence2D()

    @property
    def supported_codomain_dimensions(self) -> int:
        # If None then all codomain dimensions are supported
        return 2
