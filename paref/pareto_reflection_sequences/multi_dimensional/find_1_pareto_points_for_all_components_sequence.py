from typing import List, Optional

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.moo_algorithms.stopping_criteria.convergence_reached import ConvergenceReached
from paref.pareto_reflection_sequences.generic.next_when_stopping_criteria_met import NextWhenStoppingCriteriaMet
from paref.pareto_reflections.find_1_pareto_points import Find1ParetoPoints


class Find1ParetoPointsForAllComponentsSequence(SequenceParetoReflections):
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
    # TBA: add

    """

    def __init__(self, epsilon: float = 1e-3, stopping_criteria: ConvergenceReached = ConvergenceReached()):
        self._iter = 0
        self._sequence = None
        self.epsilon = epsilon
        self.stopping_criteria = stopping_criteria
        """
        Parameters
        ----------

        epsilon : float default 1e-3
            weight on the components

        .. warning::

            The smaller epsilon, the better. However, picking an epsilon too small may lead to an
            unstable optimization.
        """

    def next(self, blackbox_function: BlackboxFunction) -> Optional[ParetoReflection]:
        dimension_domain = blackbox_function.dimension_target_space
        pareto_reflections = [Find1ParetoPoints(blackbox_function=blackbox_function, dimension=i, epsilon=self.epsilon)
                              for i in range(dimension_domain)]
        if self._iter == 0:
            self._sequence = NextWhenStoppingCriteriaMet(pareto_reflections=pareto_reflections,
                                                         stopping_criteria=self.stopping_criteria)
            self._iter = 1

        return self._sequence.next(blackbox_function)
