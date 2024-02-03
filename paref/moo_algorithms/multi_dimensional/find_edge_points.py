from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer
from paref.pareto_reflection_sequences.multi_dimensional.find_edge_points_sequence import FindEdgePointsSequence


class FindEdgePoints(GPRMinimizer):
    """Find edge points of Pareto front

    Use this algorithm if you want to find the edge points of the Pareto front.

    """

    @property
    def sequence_of_pareto_reflections(self) -> SequenceParetoReflections:
        return FindEdgePointsSequence()

    @property
    def supported_codomain_dimensions(self) -> None:
        return None
