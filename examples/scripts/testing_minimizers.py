from typing import Union

import numpy as np
from scipy.stats import qmc

from paref.black_box_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.moo_algorithms.paref_moo import ParefMOO
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached


def function(x: np.ndarray):
    return np.sum(x ** 2)


class TestFunction(BlackboxFunction):
    def __init__(self):
        super().__init__()
        # LH evaluation
        [self(x) for x in qmc.scale(
            qmc.LatinHypercube(d=self.dimension_design_space).random(n=20),
            self.design_space.lower_bounds,
            self.design_space.upper_bounds,
        )]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.array([function(x)])
        self._evaluations.append([x, y])
        return y

    @property
    def dimension_design_space(self) -> int:
        return 5

    @property
    def dimension_target_space(self) -> int:
        return 1

    @property
    def design_space(self) -> Union[Bounds]:
        return Bounds(upper_bounds=np.ones(self.dimension_design_space),
                      lower_bounds=-np.ones(self.dimension_design_space))


class TestingMinimizers:
    def __init__(self, ):
        self.stopping_criteria = MaxIterationsReached(max_iterations=1)

    def __call__(self, moo: ParefMOO):
        blackbox_function = TestFunction()
        moo(blackbox_function, stopping_criteria=self.stopping_criteria)
        print(f"Found minimum: {np.min(blackbox_function.y)} compared to true minimum {0}")
