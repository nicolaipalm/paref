from typing import Optional, Union

import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.pareto_reflections.avoid_points import AvoidPoints


class AvoidParetoFront(SequenceParetoReflections):
    """Interface for sequences of Pareto reflections

    Documentation of an implementation of this interface should contain:

    When to use
    -----------
    This Pareto reflection should be used if...

    What it does
    ------------
    The Pareto points of this map are...

    Mathematical formula
    --------------------

    Examples
    --------

    #TODO: this should be in a contributing.md
    """
    def __init__(self,nadir: np.ndarray,
                 epsilon: Union[float, np.ndarray]):
        self.nadir = nadir
        self.epsilon = epsilon

    def next(self, blackbox_function: BlackboxFunction) -> ParetoReflection:
        """

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            The blackbox function to which the algorithm is applied

        Returns
        -------
        Optional[ParetoReflection]
            Either the next Pareto reflection or None if the end of the sequence is reached
        """
        return AvoidPoints(nadir=self.nadir,epsilon_avoiding_points=blackbox_function.pareto_front,epsilon=self.epsilon)
