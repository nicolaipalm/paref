from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer
from paref.pareto_reflection_sequences.multi_dimensional.find_1_pareto_points_for_all_components_sequence import \
    Find1ParetoPointsForAllComponentsSequence


class Find1ParetoPoints(GPRMinimizer):
    """Find 1 Pareto points



    .. note::

        Use this algorithm if you want to find a 1 Pareto point (i.e. a minimum in some component)
        for each component, f.e. in order to estimate the dimension of the Pareto front.


    Examples
    --------
    # TBA: add


    """
    @property
    def sequence_of_pareto_reflections(self) -> SequenceParetoReflections:
        return Find1ParetoPointsForAllComponentsSequence()

    @property
    def supported_codomain_dimensions(self) -> None:
        return None
