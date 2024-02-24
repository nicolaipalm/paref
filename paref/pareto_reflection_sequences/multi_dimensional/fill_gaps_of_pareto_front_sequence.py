import itertools
import warnings
import time

import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import \
    SequenceParetoReflections
from paref.pareto_reflections.fill_gap import FillGap


class FillGapsOfParetoFrontSequence(SequenceParetoReflections):
    """Close the gaps in the previously determined Pareto front

    .. warning::

        This sequence is still under development and might not work properly.

    When to use
    -----------
    Use this sequence if you want to successively close the gaps in the approximate Pareto front (the Pareto front of
    the evaluations).
    This means Pareto points are searched which lie in the *center* of gap spanning Pareto points.

    What it does
    ------------
    This sequence returns in each step the
    :py:class:`fill gap <paref.pareto_reflections.fill_gap.FillGap>`
    Pareto reflection with gap spanning points corresponding to the greatest gap in the Pareto front of the evaluations.
    """

    def __init__(self,
                 ):
        warnings.warn('This sequence is still under development and might not work properly!')
        time.sleep(0.1)

    def next(self, blackbox_function: BlackboxFunction) -> FillGap:
        """Return a :py:class:`fill gap <paref.pareto_reflections.fill_gap_2d.FillGap2D>` Pareto reflection

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            blackbox function to which sequence is applied

        Returns
        -------
        FillGap
            fill gap Pareto reflection corresponding to the greatest gap in Pareto front of the evaluations

        """
        pareto_front = np.unique(blackbox_function.pareto_front, axis=0)

        # TODO: bug - max min distance to centers does not return greatest gap;
        #  need projection to hyperplane and then max min
        # calculate (overlapping) simplices
        simplices = np.array(
            list(itertools.combinations(range(len(pareto_front)), blackbox_function.dimension_target_space)))

        # calculate centers of simplices
        centers = np.array(
            [np.sum(pareto_front[simplex], axis=0) / blackbox_function.dimension_target_space for simplex in simplices])

        # calculate distances of points to centers
        min_distances_to_centers = np.array(
            [np.min(np.linalg.norm(pareto_front - center, axis=1)) for center in centers])
        index = np.argmax(min_distances_to_centers)

        return FillGap(blackbox_function=blackbox_function,
                       gap_points=pareto_front[simplices[index]],
                       )
