from paref.moo_algorithms.stopping_criteria.convergence_reached import ConvergenceReached
from paref.pareto_reflection_sequences.generic.next_when_stopping_criteria_met import NextWhenStoppingCriteriaMet
from paref.pareto_reflections.find_one_pareto_points import FindOneParetoPoints


class FindEdgePointsSequence2D(NextWhenStoppingCriteriaMet):
    def __init__(self, ):
        super().__init__(pareto_reflections=[FindOneParetoPoints(dimension_domain=2, dimension=i) for i in range(2)],
                         stopping_criteria=ConvergenceReached())
