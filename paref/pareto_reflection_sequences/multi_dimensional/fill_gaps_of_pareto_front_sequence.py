import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import \
    SequenceParetoReflections
from paref.pareto_reflections.fill_gap import FillGap


class FillGapsOfParetoFrontSequence(SequenceParetoReflections):
    """Fill the gaps in the Pareto front already found

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

    Examples
    --------
    # TBA: add

    """

    def __init__(self,
                 epsilon: float = 0.01
                 ):
        """Specify some utopia point

        Parameters
        ----------
        epsilon : float default 0.01
            epsilon for numerical stability

        """
        self.epsilon = epsilon

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
        PF = blackbox_function.pareto_front
        # Sort Pareto points ascending by first component
        PF = PF[PF[:, 0].argsort()]

        # Calculate points with maximal distance
        max_norm_index = np.argmax(np.linalg.norm(PF[:-1] - PF[1:], axis=1))

        return FillGap(dimension_domain=blackbox_function.dimension_target_space,
                       gap_points=np.array([PF[max_norm_index], PF[max_norm_index + 1]]),
                       epsilon=self.epsilon
                       )
