from typing import Optional

import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import \
    SequenceParetoReflections
from paref.pareto_reflections.fill_gap_2d import FillGap2D


class FillGapsOfParetoFrontSequence2D(SequenceParetoReflections):
    """Fill the gaps in the Pareto front already found

    .. warning::

        If the approximate Pareto front is far away from the true Pareto front, then, this algorithm might
        tries to get closer to the true Pareto front in advance of filling the gaps.
        In order to prevent this, it is recommended to find the
        edge points of the true Pareto front in advance.
        To do so, you could use
        :py:class:`fill gap
        <paref.pareto_reflection_sequences.two_dimension.find_edge_points_sequence_2d.FindEdgePointsSequence2D>`
        for example.

    When to use
    -----------
    Use this sequence if you want to successively close the gaps in the approximate Pareto front (the Pareto front of
    the evaluations).
    This means Pareto points are searched which lie in the *middle* of two Pareto points.

    What it does
    ------------
    This sequence returns in each step the
    :py:class:`fill gap <paref.pareto_reflections.fill_gap_2d.FillGap2D>`
    Pareto reflection with two points corresponding to the greatest gap in the Pareto front of the evaluations.

    Examples
    --------
    # TBA: add

    """

    def __init__(self,
                 utopia_point: Optional[np.ndarray] = None,
                 potency: int = 6,
                 ):
        """Specify some utopia point

        Parameters
        ----------
        utopia_point : Optional[np.ndarray] default None
            optional utopia point

        potency : int default 6
            potency of underlying weighted norm
        """
        self.utopia_point = utopia_point
        self.potency = potency

    def next(self, blackbox_function: BlackboxFunction) -> FillGap2D:
        """Return a :py:class:`fill gap <paref.pareto_reflections.fill_gap_2d.FillGap2D>` Pareto reflection

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            blackbox function to which sequence is applied

        Returns
        -------
        FillGap2D
            fill gap Pareto reflection corresponding to greatest gap in Pareto front of evaluations

        """
        if blackbox_function.dimension_target_space != 2:
            raise ValueError(
                f'Dimension of target space must be 2! Dimension is {blackbox_function.dimension_target_space}.')

        PF = blackbox_function.pareto_front
        # Sort Pareto points ascending by first component
        PF = PF[PF[:, 0].argsort()]

        # Calculate points with maximal distance
        max_norm_index = np.argmax(np.linalg.norm(PF[:-1] - PF[1:], axis=1))

        # If utopia point is not given, set it as lower edge of points where gap should be closed
        if self.utopia_point is None:
            utopia_point = np.min(np.array(PF[max_norm_index:max_norm_index + 2]), axis=0)

        else:
            utopia_point = self.utopia_point

        return FillGap2D(point_1=PF[max_norm_index + 1],
                         point_2=PF[max_norm_index],
                         utopia_point=utopia_point,
                         potency=self.potency)
