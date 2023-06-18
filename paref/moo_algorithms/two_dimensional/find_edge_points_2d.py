from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer
from paref.pareto_reflection_sequences.two_dimensional.find_edge_points_sequence_2d import FindEdgePointsSequence2D


class FindEdgePoints2D(GPRMinimizer):
    @property
    def sequence_of_pareto_reflections(self) -> SequenceParetoReflections:
        return FindEdgePointsSequence2D()

    @property
    def supported_codomain_dimensions(self) -> None:
        # If None then all codomain dimensions are supported
        return 2
