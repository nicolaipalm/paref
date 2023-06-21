from paref.moo_algorithms.stopping_criteria.convergence_reached import ConvergenceReached
from paref.pareto_reflection_sequences.generic.next_when_stopping_criteria_met import NextWhenStoppingCriteriaMet
from paref.pareto_reflections.find_edge_points import FindEdgePoints


class Find1ParetoPointsForEachComponentSequence(NextWhenStoppingCriteriaMet):
    """Find all edge points of the Pareto front

    ..warning::

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
    # TODO: add

    """

    def __init__(self, dimension_domain: int):
        super().__init__(
            pareto_reflections=[FindEdgePoints(dimension_domain=dimension_domain, dimension=i) for i in range(2)],
            stopping_criteria=ConvergenceReached())
