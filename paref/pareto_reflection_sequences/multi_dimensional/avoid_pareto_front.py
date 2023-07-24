from typing import Union

import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.pareto_reflections.avoid_points import AvoidPoints


class AvoidParetoFront(SequenceParetoReflections):
    """Avoid the Pareto front of the evaluations plus some epsilon

    .. warning::

        This sequence should only be used if the Pareto points found in each evaluation of the moo
        are too close together (since it may cause stability issues in the optimization).
        Then, this sequence is meant to be used as sequence of Pareto reflections parameter in the
        :py:meth:`apply to sequence method
        <paref.interfaces.moo_algorithms.paref_moo.ParefMOO.apply_to_sequence>`.
        .

    When to use
    -----------
    This Pareto reflection should be used if you want to ensure that the Pareto points found by the optimizer are have a
    certain distance to the already evaluated Pareto front.

    What it does
    ------------
    Returns in each step an instance of the
    :py:meth:`avoid points <paref.pareto_reflections.avoid_points>` Pareto reflection where the points to be avoided
    are the Pareto points of the evaluations.

    Examples
    --------

    #TBA: add
    """

    def __init__(self, nadir: np.ndarray, epsilon: Union[float, np.ndarray]):
        self.nadir = nadir
        self.epsilon = epsilon

    def next(self, blackbox_function: BlackboxFunction) -> AvoidPoints:
        """

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            The blackbox function to which the sequence is applied

        Returns
        -------
        AvoidPoints
            avoid points Pareto reflection w.r.t. the Pareto front of the evaluations

        """
        return AvoidPoints(nadir=self.nadir,
                           epsilon_avoiding_points=blackbox_function.pareto_front,
                           epsilon=self.epsilon)
