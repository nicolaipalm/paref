from typing import Optional

import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import \
    SequenceParetoReflections
from paref.pareto_reflections.fill_gap_2d import FillGap2D


class FillGapsOfParetoFrontSequence2D(SequenceParetoReflections):
    # CAUTION: those are not the edge points if dimension is greater 2!!!
    def __init__(self,
                 utopia_point: Optional[np.ndarray] = None,
                 potency: int = 6,
                 ):
        self.utopia_point = utopia_point
        self.potency = potency

    def next(self, blackbox_function: BlackboxFunction) -> ParetoReflection:
        if blackbox_function.dimension_target_space != 2:
            raise ValueError('Dimension of target space must be 2!')

        if self.utopia_point is None:
            self.utopia_point = np.min(blackbox_function.y, axis=1)

        PF = blackbox_function.pareto_front
        # Sort Pareto points ascending by first component
        PF = PF[PF[:, 0].argsort()]

        # Calculate points with maximal distance
        max_norm_index = np.argmax(np.linalg.norm(PF[:-1] - PF[1:], axis=1))
        print(PF[max_norm_index + 1], PF[max_norm_index])

        return FillGap2D(point_1=PF[max_norm_index + 1],
                         point_2=PF[max_norm_index],
                         utopia_point=self.utopia_point,
                         potency=self.potency)
