from typing import Optional
import numpy as np

from paref.interfaces.moo_algorithms.moo_algorithm import MOOAlgorithm
from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.optimizers.minimizer import Minimizer
from paref.optimizers.minimizers.differential_evolution import DifferentialEvolution
from paref.optimizers.gpr_minimizer import GPRMinimizer
from paref.pareto_reflections.weighted_norm_to_utopia import WeightedNormToUtopia
from paref.sequences_pareto_reflections.repeating_sequence import RepeatingSequence
from paref.sequences_pareto_reflections.restricting_sequence import RestrictingSequence
from paref.optimizers.stopping_criteria.max_iterations_reached import MaxIterationsReached


class EvenlyScanned2d(MOOAlgorithm):
    def __init__(self,
                 upper_bounds_x: np.ndarray,
                 lower_bounds_x: np.ndarray,
                 max_evaluations_moo: int = 20,
                 scalar: Optional[np.ndarray] = None,
                 epsilon: float = 1e-2,
                 restricting_point_wrt_previous_evaluated_point: bool = True,
                 training_iter: int = 2000,
                 min_distance_to_evaluated_points: float = 2e-2,
                 max_iter_minimizer: int = 100,
                 minimizer: Minimizer = DifferentialEvolution()):
        self._upper_bounds_x = upper_bounds_x
        self._lower_bounds_x = lower_bounds_x
        self._max_evaluations_moo = max_evaluations_moo
        self._scalar = scalar
        self._min_distance_to_evaluated_points = min_distance_to_evaluated_points
        self._minimizer = minimizer
        self._epsilon = epsilon
        self._restricting_point_wrt_previous_evaluated = restricting_point_wrt_previous_evaluated_point
        self._training_iter = training_iter
        self._max_iter_minimizer = max_iter_minimizer

    def __call__(self,
                 blackbox_function: BlackboxFunction,
                 ):
        dimension_codomain = len(blackbox_function.y[0])
        if dimension_codomain != 2:
            raise ValueError("Dimension of codomain is not 2.")

        self._nadir = np.max(blackbox_function.y) * np.ones(dimension_codomain)

        self._utopia_point = np.zeros(dimension_codomain)

        if len(blackbox_function.evaluations) == 0:
            raise ValueError("Need at least one initial evaluation of the blackbox function.")

        # find out where more points are
        moo = GPRMinimizer(minimizer=self._minimizer,
                           max_iter_minimizer=self._max_iter_minimizer,
                           training_iter=self._training_iter,
                           upper_bounds=self._upper_bounds_x,
                           min_distance_to_evaluated_points=self._min_distance_to_evaluated_points,
                           lower_bounds=self._lower_bounds_x)
        # Search for 1 Pareto points
        print("Search for 1 Pareto points...\n")
        one_pareto_points = []
        for i in range(dimension_codomain):
            scalar = np.ones(dimension_codomain)
            scalar[i] = self._epsilon
            pareto_reflecting_function = WeightedNormToUtopia(utopia_point=self._utopia_point,
                                                              potency=np.ones(dimension_codomain),
                                                              scalar=scalar)
            sequence = RepeatingSequence(pareto_reflecting_functions=[pareto_reflecting_function],
                                         stopping_criteria=MaxIterationsReached(max_iterations=1))

            moo(blackbox_function=blackbox_function,
                pareto_reflecting_sequence=sequence,
                stopping_criteria=MaxIterationsReached(max_iterations=3))

            index_one_pareto_point = np.argmin([pareto_reflecting_function(y) for y in blackbox_function.y])
            one_pareto_points.append(blackbox_function.y[index_one_pareto_point])

        # Search for evenly scanned front
        pareto_point_corresponding_to_first_component = one_pareto_points[1]
        pareto_point_corresponding_to_second_component = one_pareto_points[0]
        distance_one_pareto_points = np.abs(
            pareto_point_corresponding_to_first_component - pareto_point_corresponding_to_second_component)

        # check on which side restricting leads to more already evaluated points which are not mapped to the nadir
        side = int(np.sum(
            blackbox_function.y.T[1] <= pareto_point_corresponding_to_first_component[1]) > np.sum(
            blackbox_function.y.T[0] <= pareto_point_corresponding_to_second_component[0]))

        if side:
            restricting_point = np.array(
                [np.max(blackbox_function.y), pareto_point_corresponding_to_first_component[1]])
            scalar = np.ones(dimension_codomain)
            scalar[1] = self._epsilon
            pareto_reflecting_function = WeightedNormToUtopia(utopia_point=self._utopia_point,
                                                              potency=np.ones(dimension_codomain),
                                                              scalar=scalar)

        else:
            restricting_point = np.array(
                [pareto_point_corresponding_to_second_component[0], np.max(blackbox_function.y)])
            scalar = np.ones(dimension_codomain)
            scalar[0] = self._epsilon
            pareto_reflecting_function = WeightedNormToUtopia(utopia_point=self._utopia_point,
                                                              potency=np.ones(dimension_codomain),
                                                              scalar=scalar)

        number_remaining_evaluations = self._max_evaluations_moo - 2
        print("Search for evenly distributed Pareto points by restricting...\n")
        for _ in range(number_remaining_evaluations):
            restricting_point[side] -= 0.8 * distance_one_pareto_points[side] / (number_remaining_evaluations + 1)

            sequence = RestrictingSequence(nadir=self._nadir,
                                           restricting_point=restricting_point,
                                           pareto_reflecting_function=pareto_reflecting_function)

            moo(blackbox_function=blackbox_function,
                pareto_reflecting_sequence=sequence,
                stopping_criteria=MaxIterationsReached(max_iterations=1))

            if self._restricting_point_wrt_previous_evaluated:
                restricting_point[side] = blackbox_function.y[-1][side]

    @property
    def name(self) -> str:
        return "EvenlyScanned2d"
