import numpy as np

from paref.express.interfaces.moo_express import MOOExpress
from paref.function_library.interfaces.function import Function
from paref.moos.minimizers.interfaces.minimizer import Minimizer
from paref.moos.minimizers.differential_evolution import DifferentialEvolution
from paref.moos.gpr_minimizer import GPRMinimizer
from paref.pareto_reflecting_library.functions.weighted_norm_to_utopia import WeightedNormToUtopia
from paref.pareto_reflecting_library.sequences.repeating_sequence import RepeatingSequence
from paref.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.moos.helper_functions.return_pareto_front import return_pareto_front


class FillGaps2d(MOOExpress):
    def __init__(self,
                 upper_bounds_x: np.ndarray,
                 lower_bounds_x: np.ndarray,
                 max_evaluations_moo: int = 20,
                 training_iter: int = 2000,
                 min_distance_to_evaluated_points: float = 2e-2,
                 max_iter_minimizer: int = 100,
                 potency: int = 2,
                 minimizer: Minimizer = DifferentialEvolution()):
        self._upper_bounds_x = upper_bounds_x
        self._lower_bounds_x = lower_bounds_x
        self._max_evaluations_moo = max_evaluations_moo
        self._min_distance_to_evaluated_points = min_distance_to_evaluated_points
        self._minimizer = minimizer
        self._training_iter = training_iter
        self._max_iter_minimizer = max_iter_minimizer
        self._potency = potency

    def __call__(self,
                 blackbox_function: Function,
                 ):
        dimension_codomain = len(blackbox_function.y[0])
        if dimension_codomain != 2:
            raise ValueError("Dimension of codomain is not 2.")

        if len(blackbox_function.evaluations) == 0:
            raise ValueError("Need at least one initial evaluation of the blackbox function.")

        moo = GPRMinimizer(minimizer=self._minimizer,
                           max_iter_minimizer=self._max_iter_minimizer,
                           training_iter=self._training_iter,
                           upper_bounds=self._upper_bounds_x,
                           min_distance_to_evaluated_points=self._min_distance_to_evaluated_points,
                           lower_bounds=self._lower_bounds_x)

        for _ in range(self._max_evaluations_moo):
            # Determine Pareto front
            PF = return_pareto_front(blackbox_function.y)

            # Sort Pareto points ascending by first component
            PF = PF[PF[:, 0].argsort()]

            # Calculate points with maximal distance
            min_norm_index = np.argmax(np.linalg.norm(PF[:-1] - PF[1:], axis=1))

            # Calculate utopia point: utopia point is given by middle point where re
            utopia_point = 1 / 2 * (PF[min_norm_index + 1] + PF[min_norm_index])
            if PF[min_norm_index][1] - PF[min_norm_index + 1][1] <= PF[min_norm_index + 1][0] - PF[min_norm_index][0]:
                utopia_point[1] = PF[min_norm_index + 1][1]
            else:
                utopia_point[0] = PF[min_norm_index][0]
            print(utopia_point)

            # Search for Pareto point in of the greatest gap of current Pareto points
            pareto_reflecting_function = WeightedNormToUtopia(
                utopia_point=utopia_point,
                potency=self._potency * np.ones(dimension_codomain),
                scalar=np.ones(dimension_codomain))
            sequence = RepeatingSequence(pareto_reflecting_functions=[pareto_reflecting_function],
                                         stopping_criteria=MaxIterationsReached(max_iterations=1))

            moo(blackbox_function=blackbox_function,
                pareto_reflecting_sequence=sequence,
                stopping_criteria=MaxIterationsReached(max_iterations=1))

    @property
    def name(self) -> str:
        return "FillGaps2d"
