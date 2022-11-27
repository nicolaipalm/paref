from typing import Optional
import numpy as np

from weimoo.express.interfaces.moo_express import MOOExpress
from weimoo.function_library.interfaces.function import Function
from weimoo.minimizers.interfaces.minimizer import Minimizer
from weimoo.minimizers.differential_evolution import DifferentialEvolution
from weimoo.moos.gpr_minimizer import GPRMinimizer
from weimoo.pareto_reflecting_library.functions.weighted_norm_to_utopia import WeightedNormToUtopia
from weimoo.pareto_reflecting_library.sequences.repeating_sequence import RepeatingSequence
from weimoo.stopping_criteria.convergence_reached import ConvergenceReached
from weimoo.stopping_criteria.interfaces.logical_or_stopping_criteria import LogicalOrStoppingCriteria
from weimoo.stopping_criteria.max_iterations_reached import MaxIterationsReached


class WeightedNormToUtopiaGPR(MOOExpress):
    def __init__(self,
                 upper_bounds_x: np.ndarray,
                 lower_bounds_x: np.ndarray,
                 max_evaluations_moo: int = 20,
                 epsilon: float = 1e-3,
                 utopia_point: Optional[np.ndarray] = None,
                 potency: Optional[np.ndarray] = None,
                 scalar: Optional[np.ndarray] = None,
                 minimizer: Minimizer = DifferentialEvolution()):
        self._upper_bounds_x = upper_bounds_x
        self._lower_bounds_x = lower_bounds_x
        self._max_evaluations_moo = max_evaluations_moo
        self._utopia_point = utopia_point
        self._potency = potency
        self._scalar = scalar
        self._minimizer = minimizer
        self._epsilon = epsilon

    def __call__(self,
                 blackbox_function: Function,
                 ):
        dimension_codomain = len(blackbox_function.y[0])
        if self._utopia_point is None:
            self._utopia_point = np.zeros(dimension_codomain)

        if self._potency is None:
            self._potency = 2 * np.ones(dimension_codomain)

        if self._scalar is None:
            self._scalar = np.ones(dimension_codomain)

        if self._upper_bounds_x is None:
            self._scalar = np.ones(dimension_codomain)

        pareto_reflecting_functions = [WeightedNormToUtopia(utopia_point=self._utopia_point,
                                                            potency=self._potency,
                                                            scalar=self._scalar),
                                       ]

        if len(blackbox_function.evaluations) == 0:
            raise ValueError("Need at least one initial evaluations of the blackbox functions.")

        stopping_criteria = LogicalOrStoppingCriteria(MaxIterationsReached(max_iterations=self._max_evaluations_moo),
                                                      ConvergenceReached(
                                                          epsilon=self._epsilon))

        sequence = RepeatingSequence(pareto_reflecting_functions=pareto_reflecting_functions,
                                     stopping_criteria=MaxIterationsReached(max_iterations=self._max_evaluations_moo))

        moo = GPRMinimizer(minimizer=self._minimizer,
                           upper_bounds=self._upper_bounds_x,
                           lower_bounds=self._lower_bounds_x)

        moo(blackbox_function=blackbox_function,
            pareto_reflecting_sequence=sequence,
            stopping_criteria=stopping_criteria)
