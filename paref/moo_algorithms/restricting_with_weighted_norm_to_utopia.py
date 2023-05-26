from typing import Optional
import numpy as np

from paref.interfaces.moo_algorithms.moo_algorithm import MOOAlgorithm
from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.optimizers.minimizer import Minimizer
from paref.optimizers.minimizers.differential_evolution import DifferentialEvolution
from paref.optimizers.gpr_minimizer import GPRMinimizer
from paref.pareto_reflections.weighted_norm_to_utopia import WeightedNormToUtopia
from paref.sequences_pareto_reflections.restricting_sequence import RestrictingSequence
from paref.sequences_pareto_reflections.stopping_criteria.max_iterations_reached import MaxIterationsReached


class RestrictingWithWeightedNormToUtopia(MOOAlgorithm):
    def __init__(self,
                 upper_bounds_x: np.ndarray,
                 lower_bounds_x: np.ndarray,
                 nadir: np.ndarray,
                 restricting_point: np.ndarray,
                 max_evaluations_moo: int = 20,
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
        self._restricting_point = restricting_point
        self._nadir = nadir

    def __call__(self,
                 blackbox_function: BlackboxFunction,
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

        pareto_reflecting_function = WeightedNormToUtopia(utopia_point=self._utopia_point,
                                                          potency=self._potency,
                                                          scalar=self._scalar)

        if len(blackbox_function.evaluations) == 0:
            raise ValueError('Need at least one initial evaluation of the blackbox blackbox_function.')

        stopping_criteria = MaxIterationsReached(max_iterations=self._max_evaluations_moo)

        sequence = RestrictingSequence(nadir=self._nadir,
                                       stopping_criteria=stopping_criteria,
                                       pareto_reflecting_function=pareto_reflecting_function,
                                       restricting_point=self._restricting_point)

        moo = GPRMinimizer(minimizer=self._minimizer,
                           upper_bounds=self._upper_bounds_x,
                           lower_bounds=self._lower_bounds_x)

        moo(blackbox_function=blackbox_function,
            pareto_reflecting_sequence=sequence,)

    @property
    def name(self) -> str:
        return 'RestrictingWithWeightedNormToUtopia'