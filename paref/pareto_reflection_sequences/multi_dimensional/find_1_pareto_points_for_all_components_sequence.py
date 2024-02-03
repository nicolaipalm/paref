from typing import Optional

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.pareto_reflection_sequences.generic.repeating_sequence import RepeatingSequence
from paref.pareto_reflections.find_1_pareto_points import Find1ParetoPoints


class Find1ParetoPointsForAllComponentsSequence(SequenceParetoReflections):
    """Find a one Pareto point for each component

    When to use
    -----------
    Use this sequence if you want to determine a one Pareto point (i.e. a Pareto point which is minimal
    in some component) of your Pareto front for each component,
    e.g. if you want to know the size of the Pareto front.

    What it does
    ------------
    This sequence applies the
    :py:class:`search for one Pareto points <paref.pareto_reflections.find_one_pareto_points.Find1ParetoPoints>`
    Pareto reflection to all components until the search converges.
    Notice: *Only* in two dimensions, the one Pareto points are precisely the edge points.
    """

    def __init__(self,):
        self._iter = 0
        self._sequence = None

    def next(self, blackbox_function: BlackboxFunction) -> Optional[ParetoReflection]:
        dimension_domain = blackbox_function.dimension_target_space
        pareto_reflections = [Find1ParetoPoints(dimension=i, blackbox_function=blackbox_function)
                              for i in range(dimension_domain)]
        if self._iter == 0:
            self._sequence = RepeatingSequence(pareto_reflections=pareto_reflections)
            self._iter = 1

        return self._sequence.next(blackbox_function)
