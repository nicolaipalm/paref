from typing import Optional

import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.moo_algorithms.minimizer.gpr_minimizer import DifferentialEvolution
from paref.moo_algorithms.minimizer.surrogates.gpr import GPR


class GprBbf(BlackboxFunction):
    def __init__(self, bbf: BlackboxFunction, training_iter=2000, learning_rate=0.01):
        self._bbf = bbf
        self._evaluations = []
        self._gpr = GPR(training_iter=training_iter, learning_rate=learning_rate)
        self._gpr.train(train_x=self.scaling_input(self.data.inputs), train_y=self.scaling_output(self.data.outputs))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = self._gpr(x)
        self._evaluations.append([x, y])
        return y

    @property
    def dimension_design_space(self) -> int:
        return self._bbf.dimension_design_space

    @property
    def dimension_target_space(self) -> int:
        return self._bbf.dimension_target_space

    @property
    def design_space(self):
        return self._bbf.design_space


class Info:
    # Purpose: information about Pareto front

    def __init__(self, blackbox_function: BlackboxFunction, training_iter=2000, learning_rate=0.01):
        self._blackbox_function = blackbox_function
        self._surrogate = GprBbf(blackbox_function, training_iter, learning_rate)
        self._minimizer = DifferentialEvolution()

        # is there a trade off ?!!! -> print: yes there is a trade off or there is a global optimum

        # say how much better this point is than evaluations so far

        # with explanation what concav etc means

        print("You can access ... by ...")

    @property
    def topology(self):
        # min max, is there a trade off, convex concave, dimensionality -> trade off only in one component
        # explain what this suggests
        pass

    def suggest(self, ranking: Optional, evaluate: bool = False):
        # answer what and why -> knee point; max Pareto point or by ranking; maybe there is a global optimum
        input("Should I evaluate at that point?")
        pass

        if isinstance(blackbox_function.design_space, Bounds):
            res = self._minimizer(
                function=fun,
                max_iter=self._max_iter_minimizer,
                upper_bounds=blackbox_function.design_space.upper_bounds,
                lower_bounds=blackbox_function.design_space.lower_bounds,
            )

        else:
            raise ValueError('Design space property of blackbox function must be an instance of Bounds!')
