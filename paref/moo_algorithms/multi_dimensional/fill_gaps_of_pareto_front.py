from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer
from paref.pareto_reflection_sequences.multi_dimensional.fill_gaps_of_pareto_front_sequence import \
    FillGapsOfParetoFrontSequence


class FillGapsOfParetoFront(GPRMinimizer):
    """Fill gaps of Pareto front

    .. note::

        Use this algorithm if you want to fill the gaps between the currently found Pareto front by Pareto points
        (Pareto front of the evaluations) which are closest to the center.


    Examples
    --------
    # TBA: add


    """

    @property
    def sequence_of_pareto_reflections(self) -> SequenceParetoReflections:
        return FillGapsOfParetoFrontSequence()

    @property
    def supported_codomain_dimensions(self) -> None:
        # If None then all codomain dimensions are supported
        return None
