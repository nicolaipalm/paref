from typing import Optional, List

import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflection_sequences.multi_dimensional.find_edge_points_sequence import FindEdgePointsSequence
from paref.pareto_reflections.find_maximal_pareto_point import FindMaximalParetoPoint
from paref.pareto_reflections.operations.compose_reflections import ComposeReflections
from paref.pareto_reflections.operations.compose_sequences import ComposeSequences
from paref.pareto_reflections.priority_search import PrioritySearch
from paref.pareto_reflections.restrict_by_point import RestrictByPoint
from paref.pareto_reflections.find_1_pareto_points import Find1ParetoPoints as Find1ParetoPointsReflection


class ExpressSearch:
    """High level MOO algorithms

    Paref express provides high level MOO algorithms that are easy and robust to use. The algorithms are
    designed to cover most frequently used MOOs and to provide a good starting point for further optimization.
    Those include:

    * minimal search: Determine the edges and a maximal point of the Pareto front, i.e. Pareto points laying on the
        boundary of the Pareto front and some Pareto point which is a real trade-off between all components.
        This algorithm should always be applied prior to any other MOO algorithm.

    * priority search: Find Pareto points which reflect your priorities of certain components. With this algorithm you
        can find Pareto points which are optimal for your specific needs.

    .. warning::

        Paref's ExpressSearch is still under development.
        If you run into any problems, errors or have suggestions how to make it **user-friendlier**, please contact me
        or open an issue on GitHub. Many thanks!

    """

    def __init__(self,
                 blackbox_function: BlackboxFunction,
                 constraints: Optional[np.ndarray] = None,
                 training_iter: int = 2000, ):
        """

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            blackbox function to apply the MOO algorithm to
        constraints : np.ndarray
            constraints for the target space i.e. what the maximal acceptable value of each component is
        training_iter : int
            number of iterations for the underlying Gaussian Process Regressors
        """
        self._bbf = blackbox_function
        self._constraints = constraints
        self._edge_points = np.empty(
            (blackbox_function.dimension_target_space, blackbox_function.dimension_target_space), dtype=object)
        self._one_points = np.empty(
            (blackbox_function.dimension_target_space, blackbox_function.dimension_target_space), dtype=object)
        self._max_point = None
        self._min_g = None
        self._priority_points = []
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
        """Determine the edges and a maximal point of the Pareto front

        This algorithm provides a minimal search for the Pareto front which allow
        to get a feeling for the problem and which algorithms to use in the next step.

        Parameters
        ----------
        max_evaluations : int
            maximum number of allowed evaluations of the blackbox function

        Examples
        --------

        >>> from paref.express.express_search import ExpressSearch
        >>> moo = ExpressSearch(bbf,)
        >>> moo.minimal_search(3)  # minimal search granting 3 evaluations of the blackbox function

        """
        if max_evaluations < self._bbf.dimension_target_space + 1:
            raise ValueError(f'You must at least grand one evaluation per component of the target space plus one '
                             f'evaluation for the maximal Pareto point, i.e. max_evaluations must be at least '
                             f'{self._bbf.dimension_target_space + 1}!')

        max_evals_maximal_point = int(
            max_evaluations // (self._bbf.dimension_target_space + 1))

        max_evals_components = max_evaluations - max_evals_maximal_point

        ############################################################################################################
        # Find Pareto points minimal in some components for each component
        min_components_sequence = FindEdgePointsSequence()

        if self._constraints is not None:
            min_components_sequence = ComposeSequences(self._constraints, min_components_sequence)

        min_component_moo = GPRMinimizer(training_iter=self._training_iter, )
        min_component_moo.apply_to_sequence(self._bbf,
                                            min_components_sequence,
                                            MaxIterationsReached(max_iterations=max_evals_components))

        self._edge_points = min_component_moo.best_fits

        ############################################################################################################
        # Find some Pareto point which is a real trade-off between all components
        self.search_for_best_real_trade_off(max_evals_maximal_point)
        print("Access the edge Pareto points by calling the attribute 'edge_points'.")

    def search_for_minima(self, max_evaluations: int, component: int):
        """Find Pareto points minimal in some component

        Parameters
        ----------
        max_evaluations : int
            maximum number of allowed evaluations of the blackbox function
        component : int
            component to minimize

        Examples
        --------

        >>> from paref.express.express_search import ExpressSearch
        >>> moo = ExpressSearch(bbf,)
        >>> moo.search_for_minima(max_evaluations=3, component=0)  # search for Pareto point minimal in the 0th obj

        """
        min_component_reflection = Find1ParetoPointsReflection(dimension=component, blackbox_function=self._bbf, )

        if self._constraints is not None:
            min_component_reflection = ComposeReflections(self._constraints, min_component_reflection)

        min_component_moo = GPRMinimizer(training_iter=self._training_iter, )
        min_component_moo.apply_to_sequence(self._bbf,
                                            min_component_reflection,
                                            MaxIterationsReached(max_iterations=max_evaluations))

        self._one_points[component] = min_component_moo.best_fits
        print("Access the Pareto points minimizing some component by calling the attribute 'minima_components'.")

    def search_for_best_real_trade_off(self, max_evaluations: int):
        """Find some Pareto point which is a real trade-off between all components

        Parameters
        ----------
        max_evaluations : int
            maximum number of allowed evaluations of the blackbox function

        Examples
        --------

        >>> from paref.express.express_search import ExpressSearch
        >>> moo = ExpressSearch(bbf,)
        >>> moo.search_for_best_real_trade_off(3)  # determine real-trade off in all components with 3 evaluations

        """
        moo_max_point_reflection = FindMaximalParetoPoint(blackbox_function=self._bbf, )

        if self._constraints is not None:
            moo_max_point_reflection = ComposeReflections(self._constraints, moo_max_point_reflection)

        moo_max_point = GPRMinimizer(training_iter=self._training_iter, )
        moo_max_point.apply_to_sequence(self._bbf,
                                        moo_max_point_reflection,
                                        MaxIterationsReached(max_iterations=max_evaluations))

        self._max_point = moo_max_point.best_fits
        print("Access the best real trade-off by calling the attribute 'max_point'.")

    def priority_search(self, priority: np.ndarray, max_evaluations: int):
        """Find Pareto points which reflect your priorities of certain components

        ``priority`` must be a vector of length equal to the number of components of the target space.
        each entry represents how much weight is assigned to the corresponding component, i.e. the higher the value
        the more important the component.

        Parameters
        ----------
        priority :  np.ndarray
            priority vector
        max_evaluations : int
            maximum number of allowed evaluations of the blackbox function

        Examples
        --------

        >>> from paref.express.express_search import ExpressSearch
        >>> moo = ExpressSearch(bbf,)
        >>> moo.minimal_search(3)  # minimal search granting 3 evaluations of the blackbox function

        """
        reflection = PrioritySearch(blackbox_function=self._bbf, priority=priority)
        moo_g = GPRMinimizer(training_iter=self._training_iter, )
        moo_g.apply_to_sequence(self._bbf,
                                reflection,
                                MaxIterationsReached(max_iterations=max_evaluations))
        self._priority_points.append(moo_g.best_fits)
        print("Access the best fitting Pareto points by calling the attribute 'priority_point'.")

    @property
    def edge_points(self) -> np.ndarray:
        """Pareto points minimal in some components

        Returns
        -------
        np.ndarray
            Pareto points minimal in some components where the ith entry
            is the Pareto points minimal in the ith component
        """
        return self._edge_points

    @property
    def minima_components(self) -> np.ndarray:
        """Pareto points minimal in some components

        Returns
        -------
        np.ndarray
            Pareto points minimal in some components where the ith entry
            is the Pareto points minimal in the ith component
        """
        return self._one_points

    @property
    def max_point(self) -> np.ndarray:
        """Real trade-off closest to the theoretical global optimum

        Returns
        -------
        np.ndarray
            Real trade-off closest to the theoretical global optimum
        """
        return self._max_point

    @property
    def priority_point(self) -> List[np.ndarray]:
        """Pareto points reflecting your priorities of certain components

        Returns
        -------
        List[np.ndarray]
            Pareto points reflecting your priorities of certain components.
            The ith element corresponds to the ith priority.
        """
        return self._priority_points
