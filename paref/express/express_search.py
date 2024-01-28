from typing import Optional, Callable

import numpy as np

from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer
from paref.moo_algorithms.multi_dimensional.find_1_pareto_points import Find1ParetoPoints
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflections.find_maximal_pareto_point import FindMaximalParetoPoint
from paref.pareto_reflections.minimize_g import MinGParetoReflection
from paref.pareto_reflections.operations.compose_reflections import ComposeReflections
from paref.pareto_reflections.priority_search import PrioritySearch
from paref.pareto_reflections.restrict_by_point import RestrictByPoint


class ExpressSearch:
    # Purpose: most intuitive but not flexible Pareto front search
    """

    .. warning::

        Paref's Express search is still under development and testing.
        If you run into any problems, errors or have suggestions how to make it *user-friendlier*, please contact me
        or open an issue on GitHub. Many thanks!
    """

    def __init__(self,
                 blackbox_function,
                 constraints: Optional[np.ndarray] = None,
                 training_iter: int = 2000,
                 max_iter_minimizer: int = 100,
                 learning_rate: float = 0.05, ):
        self._bbf = blackbox_function
        self._constraints = constraints

        self._one_points = None
        self._max_point = None
        self._min_g = None
        self._priority_points = []
        self._training_iter = training_iter

        print('TIPP. Use the Info class first to get a feeling for the problem and which algorithms to use:\n'
              'from paref.info import Info\n')

        # constraints
        if constraints is not None:
            if len(constraints) != blackbox_function.dimension_target_space:
                raise ValueError(f'Constraints must have length {blackbox_function.dimension_target_space}!')
            self._constraints = RestrictByPoint(nadir=10 * blackbox_function.y.max(axis=0),
                                                restricting_point=constraints)
        else:
            self._constraints = None

    def minimal_search(self, max_evaluations: int):
        max_evals_components = int(max_evaluations // (5 / 4))
        max_evals_maximal_point = max_evaluations - max_evals_components

        moo_one_points = Find1ParetoPoints(training_iter=self._training_iter)
        moo_one_points(self._bbf,
                       MaxIterationsReached(max_iterations=max_evals_components))

        self._one_points = moo_one_points.best_fits

        moo_max_point_reflection = FindMaximalParetoPoint(blackbox_function=self._bbf, )

        if self._constraints is not None:
            moo_max_point_reflection = ComposeReflections(self._constraints, moo_max_point_reflection)

        moo_max_point = GPRMinimizer(training_iter=self._training_iter, )
        moo_max_point.apply_to_sequence(self._bbf,
                                        moo_max_point_reflection,
                                        MaxIterationsReached(max_iterations=max_evals_maximal_point))

        self._max_point = moo_max_point.best_fits

        print("Access the best fitting Pareto points by calling the attributes 'minima_components' and 'max_point'.")

    def priority_search(self, priority: np.ndarray, max_evaluations: int):
        reflection = PrioritySearch(blackbox_function=self._bbf, priority=priority)
        moo_g = GPRMinimizer(training_iter=self._training_iter, )
        moo_g.apply_to_sequence(self._bbf,
                                reflection,
                                MaxIterationsReached(max_iterations=max_evaluations))
        self._priority_points.append(moo_g.best_fits)
        print("Access the best fitting Pareto points by calling the attribute 'min_g'.")

    def minimize_g(self, g: Callable, max_evaluations: int):
        reflection = MinGParetoReflection(g=g, blackbox_function=self._bbf)
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
    def min_g(self):
        return self._min_g

    @property
    def priority_point(self):
        return self._priority_points
