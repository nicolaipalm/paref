from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer
from paref.pareto_reflection_sequences.multi_dimensional.find_edge_points_sequence import FindEdgePointsSequence


class FindEdgePoints(GPRMinimizer):
    """Find edge points of Pareto front



    .. note::

        Use this algorithm if you want to find the edge points of the Pareto front, if they exist.
        *Notice:* In two dimensions, they always exist. However, in general they are likely to not exist.
        In order to test if the algorithm performed well, it is recommended to search for 1 Pareto points first
        (f.e. by
        :py:class:`this algorithm <paref.moo_algorithms.multi_dimensional.find_1_pareto_points.Find1ParetoPoints>`.)
        and then check if the found Pareto points are really minima in several components

    Examples
    --------
    # TBA: add


    """
    @property
    def sequence_of_pareto_reflections(self) -> SequenceParetoReflections:
        return FindEdgePointsSequence()

    @property
    def supported_codomain_dimensions(self) -> None:
        return None
