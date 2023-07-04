from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer
from paref.pareto_reflection_sequences.two_dimensional.fill_gaps_of_pareto_front_sequence_2d import \
    FillGapsOfParetoFrontSequence2D


class FillGapsOfParetoFront2D(GPRMinimizer):
    """Fill gaps of Pareto front in two dimensions

    .. note::

        Use this algorithm if you want to fill (by middle point) the gaps between the currently found Pareto front
        (Pareto front of the evaluations).


    .. warning::

        The algorithm calculates an utopia point under the hood by setting it to the minimum of the components of the
        evaluations.
        If is, therefore, highly recommended to run the
        :py:class:`find edge points algorithm <paref.moo_algorithms.multi_dimensional.find_edge_points.FindEdgePoints>`
        before applying this algorithm.


    Examples
    --------
    # TBA: add


    """

    @property
    def sequence_of_pareto_reflections(self) -> SequenceParetoReflections:
        return FillGapsOfParetoFrontSequence2D()

    @property
    def supported_codomain_dimensions(self) -> int:
        # If None then all codomain dimensions are supported
        return 2
