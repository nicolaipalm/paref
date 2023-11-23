from warnings import warn
from typing import Union, Optional, Callable

import numpy as np

from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer
from paref.moo_algorithms.minimizer.surrogates.gpr import GPR
from paref.moo_algorithms.multi_dimensional.find_1_pareto_points import Find1ParetoPoints
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflections.minimize_g import MinGParetoReflection
from paref.pareto_reflections.minimize_weighted_norm_to_utopia import MinimizeWeightedNormToUtopia
from paref.pareto_reflections.operations.compose_reflections import ComposeReflections
from paref.pareto_reflections.restrict_by_point import RestrictByPoint


# join max iterations and convergence reached? -> problem: which scale
# parameter for each component -> stopping criterion
# TODO: use Info in init? -> NO! should be only algorithms; but can include as optional
# constraints: join with Pareto reflection


# TODO: include Info class to estimate how many evaluations are needed...

class ExpressSearch:
    # Purpose: most intuitive but not flexible Pareto front search
    """
    TODO: print(tipp: use the Info class first to get a feeling for the problem and which algorithms to use)
    """

    def __init__(self,
                 blackbox_function,
                 constraints: Optional[np.ndarray] = None,
                 training_iter: Union[int, str] = 'auto',
                 max_iter_minimizer: int = 100,
                 learning_rate: float = 0.05, ):
        self._bbf = blackbox_function
        self._constraints = constraints

        self._one_points = None
        self._max_point = None
        self._edge_points = None
        self._min_g = None
        self._priority_point = []

        # check if trainings iterations are sufficient
        # TODO: also include iter minimizer and learning rate
        if training_iter == 'auto':
            print('Obtaining optimal number of training iterations...')
            for i in [1000, 2000, 5000, 10000, 20000]:
                gpr = GPR(training_iter=i)
                gpr.train(train_x=blackbox_function.x, train_y=blackbox_function.y)
                if np.all(gpr.model_convergence < 0.05):
                    print(f'Optimal Training iterations: {i}')
                    gpr.plot_loss()
                    training_iter = i
                    break

            if training_iter == 'auto':
                gpr.plot_loss()
                training_iter = 5000
                warn(f'Training might not have converged. Training iter is set to {training_iter}.')

        self._training_iter = training_iter

        # constraints
        if constraints is not None:
            if len(constraints) != blackbox_function.dimension_target_space:
                raise ValueError(f'Constraints must have length {blackbox_function.dimension_target_space}!')
            self._constraints = RestrictByPoint(nadir=10 * blackbox_function.y.max(axis=0),
                                                restricting_point=constraints)
        else:
            self._constraints = None

    def minimal_search(self, max_evaluations: int):
        # TODO: find optimal parameters automatically
        # TODO: mention how well search is working
        # minima components and maximal pareto point
        # focusing on one component and real trade off between all components

        max_evals_components = max_evaluations // (4 / 5)
        max_evals_maximal_point = max_evaluations - max_evals_components

        moo_one_points = Find1ParetoPoints(training_iter=self._training_iter)
        moo_one_points(self._bbf,
                       MaxIterationsReached(max_iterations=max_evals_components))

        self._one_points = moo_one_points.best_fits

        moo_max_point_reflection = MinimizeWeightedNormToUtopia(utopia_point=self._bbf.y.min(axis=0),
                                                                scalar=np.ones(self._bbf.dimension_target_space),
                                                                potency=4)
        if self._constraints is not None:
            moo_max_point_reflection = ComposeReflections(self._constraints, moo_max_point_reflection)

        moo_max_point = GPRMinimizer(training_iter=self._training_iter, )
        moo_max_point.apply_to_sequence(self._bbf,
                                        moo_max_point_reflection,
                                        MaxIterationsReached(max_iterations=max_evals_maximal_point))

        self._max_point = moo_max_point.best_fits

        print("Access the best fitting Pareto points by calling the attributes 'minima_components' and 'max_point'.")

    def maximal_search(self):
        # edge points and close gaps ie all pareto points
        raise NotImplementedError

    def priority_search(self):
        # give percentage for priorities of components
        raise NotImplementedError

    def minimize_g(self, g: Callable, max_evaluations: int):
        reflection = MinGParetoReflection(g=g, bbf=self._bbf)
        moo_g = GPRMinimizer(training_iter=self._training_iter, )
        moo_g.apply_to_sequence(self._bbf,
                                reflection,
                                MaxIterationsReached(max_iterations=max_evaluations))
        self._min_g = moo_g.best_fits
        print("Access the best fitting Pareto points by calling the attribute 'min_g'.")

    @property
    def minima_components(self):
        return self._one_points

    @property
    def max_point(self):
        return self._max_point

    @property
    def edge_points(self):
        return self._edge_points

    @property
    def min_g(self):
        return self._min_g
