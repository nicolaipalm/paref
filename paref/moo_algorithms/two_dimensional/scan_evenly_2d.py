from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer
from paref.moo_algorithms.stopping_criteria.convergence_reached import ConvergenceReached
from paref.pareto_reflection_sequences.generic.next_when_stopping_criteria_met import NextWhenStoppingCriteriaMet
from paref.pareto_reflections.find_1_pareto_points import Find1ParetoPoints


class ScanEvenly2D(GPRMinimizer):
    @property
    def sequence_of_pareto_reflections(self):  # -> SequenceParetoReflections:
        NextWhenStoppingCriteriaMet(stopping_criteria=ConvergenceReached(),
                                    pareto_reflections=[
                                        Find1ParetoPoints(dimension_domain=2, dimension=0),
                                        Find1ParetoPoints(dimension_domain=2, dimension=1),
                                        ])
        # TBA: grid search or one pareto points plus fill gaps
        raise NotImplementedError

    @property
    def supported_codomain_dimensions(self) -> int:
        # If None then all codomain dimensions are supported
        return 2
