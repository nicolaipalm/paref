from paref.moo_algorithms.stopping_criteria.convergence_reached import ConvergenceReached
from paref.pareto_reflection_sequences.generic.next_when_stopping_criteria_met import NextWhenStoppingCriteriaMet
from paref.pareto_reflections.find_1_pareto_points import Find1ParetoPoints


class Find1ParetoPointsForEachComponentSequence(NextWhenStoppingCriteriaMet):
    """Find a one Pareto point for each component

    When to use
    -----------
    Use this sequence if you want to determine a one Pareto point (i.e. minimal in some component) of your Pareto front
    for each component, f.e. if you want to know the size of the Pareto front.

    What it does
    ------------
    This sequence applies the
    :py:class:`search for one Pareto points <paref.pareto_reflections.find_one_pareto_points.Find1ParetoPoints>`
    Pareto reflection to all components until the search converges.
    Notice: *Only* in two dimensions, the one Pareto points are precisely the edge points.

    Examples
    --------
    # TODO: add

    """

    def __init__(self, dimension_domain: int):
        super().__init__(
            pareto_reflections=[Find1ParetoPoints(dimension_domain=dimension_domain, dimension=i) for i in range(2)],
            stopping_criteria=ConvergenceReached())
