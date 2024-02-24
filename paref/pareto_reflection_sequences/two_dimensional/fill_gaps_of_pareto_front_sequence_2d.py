import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import \
    SequenceParetoReflections
from paref.pareto_reflections.fill_gap import FillGap


class FillGapsOfParetoFrontSequence2D(SequenceParetoReflections):
    """Close the gaps in the previously determined Pareto front (in two dimensions)

    .. warning::

        It is recommended to find the
        edge points of the true Pareto front in advance.
        To do so, you could use
        :py:class:`fill gap
        <paref.pareto_reflection_sequences.multi_dimension.find_edge_points_sequence.FindEdgePointsSequence>`
        for example.

    When to use
    -----------
    Use this sequence if you want to iteratively close the gaps in the approximate Pareto front (the Pareto front of
    the evaluations).
    This means Pareto points are searched which lie in the *center* of two Pareto points.

    What it does
    ------------
    This sequence returns in each step the
    :py:class:`fill gap <paref.pareto_reflections.fill_gap_2d.FillGap2D>`
    Pareto reflection with two points corresponding to the greatest gap in the Pareto front of the evaluations.
    """

    def next(self, blackbox_function: BlackboxFunction) -> FillGap:
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

        return FillGap(blackbox_function=blackbox_function,
                       gap_points=PF[max_norm_index:max_norm_index + 2],
                       )
