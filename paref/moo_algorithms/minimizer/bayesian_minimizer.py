import random
from typing import List

import numpy as np
from bayes_opt import BayesianOptimization

from paref.black_box_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.moo_algorithms.paref_moo import ParefMOO, CompositionWithParetoReflection


class Bayesian:
    def __init__(self, display=False):
        self.display = display
        self._number_evaluations_last_call = None

    def __call__(
            self,
            function,
            pbounds: dict,
            n_iter: int = 30,
    ):
        optimizer = BayesianOptimization(
            f=function,
            pbounds=pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=random.randint(0, 10000),
            allow_duplicate_points=True,
        )

        optimizer.maximize(
            init_points=10,
            n_iter=n_iter,
        )

        self.result = optimizer.max
        self._number_evaluations_last_call = len(optimizer.res)
        return optimizer.max.get('params').values()

    @property
    def number_evaluations_last_call(self):
        return self._number_evaluations_last_call


class BayesianMinimizer(ParefMOO):
    def apply_moo_operation(self, blackbox_function: BlackboxFunction) -> None:
        minimizer = Bayesian()
        underlying_blackbox_function = blackbox_function
        while isinstance(underlying_blackbox_function, CompositionWithParetoReflection):
            underlying_blackbox_function = underlying_blackbox_function._blackbox_function

        if isinstance(blackbox_function.design_space, Bounds):

            def function_wrapper(x0, **kwargs):
                x = np.array([x0, *kwargs.values()])
                return -1 * blackbox_function(x)

            pbounds = {}
            for i in range(blackbox_function.dimension_design_space):
                pbounds[f'x{i}'] = (
                    blackbox_function.design_space.lower_bounds[i], blackbox_function.design_space.upper_bounds[i])
            minimizer(
                function=function_wrapper,
                pbounds=pbounds,
            )
        else:
            raise ValueError('Design space property of blackbox function must be an instance of Bounds!')

        print('finished!')
        print('Value of blackbox: ', underlying_blackbox_function.y[-1])

    @property
    def supported_codomain_dimensions(self) -> List[int]:
        return [1]
