from typing import Optional
import numpy as np

from paref.express.interfaces.moo_express import MOOExpress
from paref.function_library.interfaces.function import Function
from paref.moos.minimizers.interfaces.minimizer import Minimizer
from paref.moos.minimizers.differential_evolution import DifferentialEvolution
from paref.moos.gpr_minimizer import GPRMinimizer
from paref.pareto_reflecting_library.functions.weighted_norm_to_utopia import WeightedNormToUtopia
from paref.pareto_reflecting_library.sequences.repeating_sequence import RepeatingSequence
from paref.pareto_reflecting_library.sequences.restricting_sequence import RestrictingSequence
from paref.stopping_criteria.max_iterations_reached import MaxIterationsReached


class EvenlyScanned2d(MOOExpress):
    def __init__(self,
                 upper_bounds_x: np.ndarray,
                 lower_bounds_x: np.ndarray,
                 max_evaluations_moo: int = 20,
                 scalar: Optional[np.ndarray] = None,
                 epsilon: float = 1e-2,
                 minimizer: Minimizer = DifferentialEvolution()):
        self._upper_bounds_x = upper_bounds_x
        self._lower_bounds_x = lower_bounds_x
        self._max_evaluations_moo = max_evaluations_moo
        self._scalar = scalar
        self._minimizer = minimizer
        self._epsilon = epsilon

    def __call__(self,
                 blackbox_function: Function,
                 ):
        dimension_codomain = len(blackbox_function.y[0])
        if dimension_codomain != 2:
            raise ValueError("Dimension of codomain is not 2.")

        self._nadir = np.max(blackbox_function.y) * np.ones(dimension_codomain)

        self._utopia_point = np.zeros(dimension_codomain)

        if len(blackbox_function.evaluations) == 0:
            raise ValueError("Need at least one initial evaluation of the blackbox function.")

        # find out where more points are

        stopping_criteria = MaxIterationsReached(max_iterations=self._max_evaluations_moo)

        moo = GPRMinimizer(minimizer=self._minimizer,
                           upper_bounds=self._upper_bounds_x,
                           lower_bounds=self._lower_bounds_x)
        # 1 Pareto points
        print("Search for 1 Pareto points...\n")
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
                stopping_criteria=MaxIterationsReached(max_iterations=1))

        # evenly scanned
        pareto_point_corresponding_to_first_component = blackbox_function.y[-1]
        pareto_point_corresponding_to_second_component = blackbox_function.y[-2]
        distance_one_pareto_points = np.linalg.norm(
            pareto_point_corresponding_to_first_component - pareto_point_corresponding_to_second_component)

        # check on which side restricting leads to more already evaluated points which are not mapped to the nadir
        side = int(np.sum(
            blackbox_function.y.T[1] <= pareto_point_corresponding_to_first_component[1]) > np.sum(
            blackbox_function.y.T[0] <= pareto_point_corresponding_to_second_component[0]))

        if side:
            restricting_point = np.array([np.max(blackbox_function.y), pareto_point_corresponding_to_first_component[1]])
            scalar = np.ones(dimension_codomain)
            scalar[1] = self._epsilon
            pareto_reflecting_function = WeightedNormToUtopia(utopia_point=self._utopia_point,
                                                              potency=np.ones(dimension_codomain),
                                                              scalar=scalar)

        else:
            restricting_point = np.array([pareto_point_corresponding_to_second_component[0], np.max(blackbox_function.y)])
            scalar = np.ones(dimension_codomain)
            scalar[0] = self._epsilon
            pareto_reflecting_function = WeightedNormToUtopia(utopia_point=self._utopia_point,
                                                              potency=np.ones(dimension_codomain),
                                                              scalar=scalar)

        number_remaining_evaluations = self._max_evaluations_moo - 2
        print("Search for evenly distributed Pareto points by restricting...\n")
        for _ in range(number_remaining_evaluations):
            restricting_point[side] -= 0.8*distance_one_pareto_points / (number_remaining_evaluations+1)

            sequence = RestrictingSequence(nadir=self._nadir,
                                           restricting_point=restricting_point,
                                           pareto_reflecting_function=pareto_reflecting_function)

            moo(blackbox_function=blackbox_function,
                pareto_reflecting_sequence=sequence,
                stopping_criteria=MaxIterationsReached(max_iterations=1))

    @property
    def name(self) -> str:
        return "EvenlyScanned2d"
