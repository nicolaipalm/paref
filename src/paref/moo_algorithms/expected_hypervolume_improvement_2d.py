import numpy as np

from paref.interfaces.moo_algorithms.moo_algorithm import MOOAlgorithm
from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.optimizers.minimizer import Minimizer
from paref.optimizers.minimizers.differential_evolution import DifferentialEvolution
from paref.optimizers.helper_functions.return_pareto_front import return_pareto_front
from paref.pareto_reflections.expected_hypervolume_2d import ExpectedHypervolume2d
from paref.optimizers.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.optimizers.surrogates.gpr import GPR


class ExpectedHypervolumeImprovement2d(MOOAlgorithm):
    def __init__(self,
                 upper_bounds_x: np.ndarray,
                 lower_bounds_x: np.ndarray,
                 reference_point: np.ndarray,
                 number_designs_lh: int = 20,
                 max_iter_minimizer: int = 100,
                 max_evaluations_moo: int = 20,
                 training_iter: int = 2000,
                 minimizer: Minimizer = DifferentialEvolution(),
                 learning_rate: float = 0.05):
        self._minimizer = minimizer
        self._upper_bounds = upper_bounds_x
        self._lower_bounds = lower_bounds_x
        self._number_designs_lh = number_designs_lh
        self._max_iter_minimizer = max_iter_minimizer
        self._training_iter = training_iter
        self._learning_rate = learning_rate
        self._max_evaluations_moo = max_evaluations_moo
        self._reference_point = reference_point

    def __call__(self,
                 blackbox_function: BlackboxFunction):
        gpr = GPR(training_iter=self._training_iter, learning_rate=self._learning_rate)

        iteration_step = 1
        stopping_criteria = MaxIterationsReached(max_iterations=self._max_evaluations_moo)
        stop = stopping_criteria(blackbox_function=blackbox_function)
        while not stop:
            train_x = blackbox_function.x

            train_y = blackbox_function.y

            pareto_front = return_pareto_front(train_y)
            print(
                f"{iteration_step} Training of the GPR...\n"
            )
            gpr.train(train_x=train_x, train_y=train_y)
            print(f"\n finished!\n")
            print(f"Starting minimization...")

            expected_hypervolume_improvement = ExpectedHypervolume2d(reference_point=self._reference_point,
                                                         pareto_front=pareto_front, gpr=gpr)

            res = self._minimizer(
                function=lambda x: expected_hypervolume_improvement(x),
                max_iter=self._max_iter_minimizer,
                upper_bounds=self._upper_bounds,
                lower_bounds=self._lower_bounds,
            )

            print(f"\n finished!\n Evaluating blackbox function...")

            blackbox_function(res)
            print(f"\n finished!")

            iteration_step += 1

            stop = stopping_criteria(blackbox_function=blackbox_function)

    @property
    def name(self) -> str:
        return "expected_hypervolume_improvement"
