# 1/3 min component difference of pareto front
# sum without utopia point
# guarantee
# randomize weights in order to dont get stuck

import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.pareto_reflections.avoid_points import AvoidPoints
from paref.pareto_reflections.minimize_weighted_norm_to_utopia import MinimizeWeightedNormToUtopia
from paref.pareto_reflections.operations.compose_reflections import ComposeReflections


class FindAllParetoPoints(SequenceParetoReflections):
    """# TBA: add and if it is reasonable?
    """

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
        nadir = np.max(blackbox_function.y, axis=0)
        epsilon = 1 / 3 * np.min(
            np.min(np.abs(blackbox_function.pareto_front[1:] - blackbox_function.pareto_front[:-1]), axis=0))
        # scalar = 1e-2 * np.ones(blackbox_function.dimension_target_space)
        # scalar[np.random.randint(low=0, high=blackbox_function.dimension_target_space - 1)] = 1

        scalar = np.random.uniform(size=blackbox_function.dimension_target_space - 1)
        scalar = np.append(scalar, np.sqrt(1 - np.sum(scalar ** 2)))

        return ComposeReflections(pareto_reflecting_function_1=AvoidPoints(
            nadir=nadir,
            epsilon_avoiding_points=blackbox_function.pareto_front,
            epsilon=epsilon),
            pareto_reflecting_function_2=MinimizeWeightedNormToUtopia(
                utopia_point=np.zeros(blackbox_function.dimension_target_space),
                potency=np.ones(blackbox_function.dimension_target_space),
                scalar=scalar))
