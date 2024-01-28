from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import \
    SequenceParetoReflections
from paref.pareto_reflection_sequences.multi_dimensional.find_edge_points_sequence import FindEdgePointsSequence
from paref.pareto_reflections.fill_gap import FillGap


class GridSearch(SequenceParetoReflections):
    """Determine an equidistant grid of Pareto points

    When to use
    -----------
    Use this sequence when you want to determine an equidistant grid of Pareto points, for example when you want to
    get a solid approximation of the Pareto front taking into account based on the maximum number of
    evaluations you granted.

    .. note::

        This sequence should be used *at the beginning* of the optimization process, i.e. before other Pareto reflection
        are applied. This is because the sequence does not take into account the Pareto front of the evaluations and
        might, therefore, return points that are already closet to or in the Pareto front.

    What it does
    ------------
    This sequence determines an equidistant grid of Pareto points including edge points for all dimensions.

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
        self._edge_point_sequence = FindEdgePointsSequence()
        self._close_gaps_sequence = None

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
        raise NotImplementedError('This method is not implemented yet.')
