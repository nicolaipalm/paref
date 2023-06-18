import numpy as np

from paref.interfaces.moo_algorithms.paref_moo import ParefMOO, CompositionWithParetoReflection
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.black_box_functions.design_space.bounds import Bounds
from paref.moo_algorithms.minimizer.surrogates.gpr import GPR

from typing import Callable
from scipy.optimize import differential_evolution

from paref.pareto_reflections.operations.compose_reflections import ComposeReflections


class DifferentialEvolution:
    def __init__(self, display=False):
        self.display = display
        self._number_evaluations_last_call = None

    def __call__(
            self,
            function: Callable,
            upper_bounds: np.ndarray,
            lower_bounds: np.ndarray,
            max_iter: int = 1000,
    ) -> np.ndarray:
        t_initial = (upper_bounds + lower_bounds) / 2
        res = differential_evolution(
            func=function,
            x0=t_initial,
            disp=self.display,
            tol=1e-5,
            bounds=[
                (lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))
            ],
            maxiter=max_iter,
        )

        self.result = res
        self._number_evaluations_last_call = res.nfev
        return res.x

    @property
    def number_evaluations_last_call(self):
        return self._number_evaluations_last_call


class GPRMinimizer(ParefMOO):
    def __init__(self,
                 max_iter_minimizer: int = 100,
                 training_iter: int = 2000,
                 learning_rate: float = 0.05,
                 min_distance_to_evaluated_points: float = 2e-2):
        self._minimizer = DifferentialEvolution()
        self._max_iter_minimizer = max_iter_minimizer
        self._training_iter = training_iter
        self._learning_rate = learning_rate
        self._min_distance_to_evaluated_points = min_distance_to_evaluated_points

    def apply_moo_operation(self,
                            blackbox_function: BlackboxFunction,
                            ):
        # TODO: when found points are too close stop!
        # TODO: control mechanism: when algo doesnt work give message and what went wrong
        gpr = GPR(training_iter=self._training_iter, learning_rate=self._learning_rate)

        base_blackbox_function = blackbox_function
        pareto_reflections = []
        while isinstance(base_blackbox_function, CompositionWithParetoReflection):
            pareto_reflections.append(base_blackbox_function._pareto_reflection)
            base_blackbox_function = base_blackbox_function._blackbox_function

        train_x = base_blackbox_function.x
        train_y = base_blackbox_function.y

        print(
            f'Training of the GPR...\n'
        )
        gpr.train(train_x=train_x, train_y=train_y)
        print('\n finished!\n')
        print('Starting minimization...')

        if len(pareto_reflections) != 0:
            pareto_reflection = pareto_reflections[0]
            for reflection in pareto_reflections[1:]:
                pareto_reflection = ComposeReflections(reflection, pareto_reflection)
            fun = lambda x: pareto_reflection(gpr(x))

        else:
            fun = lambda x: gpr(x)

        if isinstance(blackbox_function.design_space, Bounds):
            res = self._minimizer(
                function=fun,
                max_iter=self._max_iter_minimizer,
                upper_bounds=blackbox_function.design_space.upper_bounds,
                lower_bounds=blackbox_function.design_space.lower_bounds,
            )

        else:
            raise ValueError('Design space property of blackbox function must be an instance of Bounds!')

        print('finished!')
        print(
            f'Found Pareto point: \n x={res} '
            f'\n y={gpr(res)} ')

        # if np.all(pareto_reflection(gpr(res)) >= pareto_reflection(blackbox_function.y[0])):
        #    print('\nNo Pareto point was found. Algorithmic search stopped.')

        # if np.any(np.linalg.norm(gpr(res) - np.array(
        #        [gpr(x) for x in blackbox_function.x]), axis=1) <= self._min_distance_to_evaluated_points):
        #    print('\nFound Pareto point is too close to some already evaluated point.')

        print('Evaluating blackbox blackbox_function...')
        blackbox_function(res)
        print('finished!')
        print('Value of blackbox blackbox_function: ', blackbox_function.y[-1])
        print('Difference to estimation: ', gpr(res) - blackbox_function.y[-1])

    @property
    def supported_codomain_dimensions(self) -> int:
        # If None then all codomain dimensions are supported
        return 1
