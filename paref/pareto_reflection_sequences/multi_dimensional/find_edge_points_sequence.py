from typing import Optional

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.moo_algorithms.stopping_criteria.convergence_reached import ConvergenceReached
from paref.pareto_reflection_sequences.generic.next_when_stopping_criteria_met import NextWhenStoppingCriteriaMet
from paref.pareto_reflections.find_edge_points import FindEdgePoints


class FindEdgePointsSequence(SequenceParetoReflections):
    """Find all edge points of the Pareto front

    .. warning::

        This Pareto reflection assumes that there exist edge points

    When to use
    -----------
    Use this sequence if you want to determine the edge points of your Pareto front, f.e. if you want
    to know the size of the Pareto front.

    What it does
    ------------
    This sequence applies the
    :py:class:`search for one Pareto points <paref.pareto_reflections.find_edge_points.FindEdgePoints>`
    Pareto reflection to all components until the search converges.
    Notice: *Only* in two dimensions, the one Pareto points are guaranteed to exist

    Examples
    --------
    # TBA: add

    """

    def __init__(self, ):
        self._iter = 0
        self._sequence = None
        self.stopping_criteria = ConvergenceReached()

    def next(self, blackbox_function: BlackboxFunction) -> Optional[ParetoReflection]:
        dimension_domain = blackbox_function.dimension_target_space
        pareto_reflections = [FindEdgePoints(dimension_domain=dimension_domain, dimension=i) for i in
                              range(dimension_domain)]
        if self._iter == 0:
            self._sequence = NextWhenStoppingCriteriaMet(pareto_reflections=pareto_reflections,
                                                         stopping_criteria=self.stopping_criteria)
            self._iter = 1

        return self._sequence.next(blackbox_function)
