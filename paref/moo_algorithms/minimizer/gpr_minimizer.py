from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from paref.black_box_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.moo_algorithms.paref_moo import ParefMOO, CompositionWithParetoReflection
from paref.moo_algorithms.minimizer.surrogates.gpr import GPR
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
    """Minimize any function by approximating it with a GPR and minimize the (computationally cheap) GPR

    .. note::

        This minimizer should be used in setups where the blackbox function is computationally expensive and
        only a few initial samples are available.

    How it works
    ------------
    A `GPR <https://en.wikipedia.org/wiki/Kriging>`_ is trained on the evaluations of the blackbox function.
    Then, the trained GPR is minimized by a `differential evolution algorithm
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>`_.

    Furthermore, if a multi-dimensional blackbox is composed with a (scalar valued) Pareto reflection, then, for
    each component a GPR is trained and the composition of the trained GPRs with the Pareto reflection is minimized.


    .. warning::

        This minimizer requires a number of initial evaluations in order to perform well. If the number of evaluations
        is below some threshold (default: 20),
        a `latin hypercube sampling <https://en.wikipedia.org/wiki/Latin_hypercube_samplingL>`_
        is performed before the optimizer starts.
        Per construction a minimum number of 20 initial evaluations is required.
        In addition, this minimizer requires the design space to be a cube, i.e. characterized by bounds.



    """

    def __init__(self,
                 max_iter_minimizer: int = 100,
                 training_iter: int = 2000,
                 learning_rate: float = 0.05,
                 min_required_evaluations: int = 20,
                 min_distance_to_evaluated_points: float = 2e-2):
        """Initialize the algorithms hyperparameters

        Parameters
        ----------
        max_iter_minimizer : int default 100
            maximum number of iterations of the differential evolution algorithm

        training_iter : int default 2000
            maximum training iterations of the GPR(s)

        learning_rate : float default 0.05
            learning rate of the training of the GPR(s)

        min_required_evaluations : int default 20
            minimum number of evaluations required for the training (must be greater or equal than 20)

        min_distance_to_evaluated_points : float default 2e-2
            required minimum distance to already evaluated points
        """
        self._minimizer = DifferentialEvolution()
        self._max_iter_minimizer = max_iter_minimizer
        self._training_iter = training_iter
        self._learning_rate = learning_rate
        self._min_distance_to_evaluated_points = min_distance_to_evaluated_points
        self._min_required_evaluations = min_required_evaluations

    def apply_moo_operation(self,
                            blackbox_function: BlackboxFunction,
                            ) -> None:
        """Apply moo operation constructed as above

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            blackbox function to which algorithm is applied


        """
        # TBA: when found points are too close stop!
        # TBA: control mechanism: when algo doesnt work give message and what went wrong
        # TBA: monitoring: stop time, evaluations found, if training process of gpr converged, ALL WITH hints

        if not isinstance(blackbox_function.design_space, Bounds):
            raise ValueError(f'Design space of blackbox function must be an instance of by Bounds! Design space is of'
                             f'type {blackbox_function.design_space}.')

        if self._min_required_evaluations < 20:
            print(
                'WARNING: minimum number of evaluations is 20! The minimum number of evaluations is, thus, set to 20!')

        if len(blackbox_function.evaluations) < self._min_required_evaluations:
            [blackbox_function(x) for x in qmc.scale(
                qmc.LatinHypercube(d=blackbox_function.dimension_design_space).random(
                    n=self._min_required_evaluations - len(blackbox_function.evaluations)),
                blackbox_function.design_space.lower_bounds,
                blackbox_function.design_space.upper_bounds,
            )]  # add samples according to latin hypercube scheme

        gpr = GPR(training_iter=self._training_iter, learning_rate=self._learning_rate)

        base_blackbox_function = blackbox_function

        pareto_reflections = []
        while isinstance(base_blackbox_function, CompositionWithParetoReflection):
            pareto_reflections.append(base_blackbox_function._pareto_reflection)
            base_blackbox_function = base_blackbox_function._blackbox_function

        train_x = base_blackbox_function.x
        train_y = base_blackbox_function.y

        print(
            'Training of the GPR...\n'
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
        if np.any([np.all(dominated) for dominated in (gpr(res) >= blackbox_function.pareto_front)]):
            print('WARNING: Optimizer did not find a Pareto point! Blackbox function is not evaluated.')

        # if np.all(pareto_reflection(gpr(res)) >= pareto_reflection(blackbox_function.y[0])):
        #    print('\nNo Pareto point was found. Algorithmic search stopped.')

        # if np.any(np.linalg.norm(gpr(res) - np.array(
        #        [gpr(x) for x in blackbox_function.x]), axis=1) <= self._min_distance_to_evaluated_points):
        #    print('\nFound Pareto point is too close to some already evaluated point.')

        print('Evaluating blackbox function...')
        blackbox_function(res)
        print('finished!')
        print('Value of blackbox function: ', base_blackbox_function.y[-1])
        print('Difference to estimation: ', gpr(res) - base_blackbox_function.y[-1])

    @property
    def supported_codomain_dimensions(self) -> int:
        # TBA: dimensionality check and list of int
        return 1
