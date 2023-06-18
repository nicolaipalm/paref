import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import \
    SequenceParetoReflections
from paref.pareto_reflections.minimize_weighted_norm_to_utopia import MinimizeWeightedNormToUtopia


class FindOneParetoPointSequence(SequenceParetoReflections):
    # CAUTION: those are not the edge points if dimension is greater 2!!!
    def __init__(self,
                 dimension: int,
                 epsilon: float = 1e-3,
                 ):
        self.epsilon = epsilon
        self.dimension = dimension

    def next(self, blackbox_function: BlackboxFunction) -> ParetoReflection:
        dimension_domain = blackbox_function.dimension_target_space
        scalar = self.epsilon * np.ones(dimension_domain)
        scalar[self.dimension] = 1
        return MinimizeWeightedNormToUtopia(utopia_point=np.zeros(dimension_domain),
                                            potency=np.ones(dimension_domain),
                                            scalar=scalar)
