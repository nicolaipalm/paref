import numpy as np

from paref.function_library.interfaces.function import Function
from paref.moos.minimizers.interfaces.minimizer import Minimizer
from paref.moos.minimizers.differential_evolution import DifferentialEvolution
from paref.moos.interfaces.moo import MOO
from paref.pareto_reflecting_library.sequences.interfaces.sequence_pareto_reflecting_functions import \
    SequenceParetoReflectingFunctions
from paref.stopping_criteria.interfaces.stopping_criteria import StoppingCriteria
from paref.surrogates.gpr import GPR


class GPRMinimizer(MOO):
    def __init__(self,
                 upper_bounds: np.ndarray,
                 lower_bounds: np.ndarray,
                 max_iter_minimizer: int = 100,
                 training_iter: int = 2000,
                 minimizer: Minimizer = DifferentialEvolution(),
                 learning_rate: float = 0.05,
                 min_distance_to_evaluated_points: float = 2e-2):
        self._minimizer = minimizer
        self._upper_bounds = upper_bounds
        self._lower_bounds = lower_bounds
        self._max_iter_minimizer = max_iter_minimizer
        self._training_iter = training_iter
        self._learning_rate = learning_rate
        self._min_distance_to_evaluated_points = min_distance_to_evaluated_points

    def __call__(self,
                 blackbox_function: Function,
                 pareto_reflecting_sequence: SequenceParetoReflectingFunctions,
                 stopping_criteria: StoppingCriteria):

        gpr = GPR(training_iter=self._training_iter, learning_rate=self._learning_rate)

        iteration_step = 1
        stop = stopping_criteria(blackbox_function)
        while not stop:
            train_x = blackbox_function.x

            train_y = blackbox_function.y
            print(
                f"{iteration_step} Training of the GPR...\n"
            )
            gpr.train(train_x=train_x, train_y=train_y)
            print(f"\nfinished!\n")
            print(f"Starting minimization...")

            pareto_reflecting_function = pareto_reflecting_sequence.next(blackbox_function=blackbox_function)
            res = self._minimizer(
                function=lambda x: pareto_reflecting_function(gpr(x)),
                max_iter=self._max_iter_minimizer,
                upper_bounds=self._upper_bounds,
                lower_bounds=self._lower_bounds,
            )
            print("finished!")
            print(
                f"Found Pareto point: \n x={res} "
                f"\n y={gpr(res)} "
                f"\n value at Pareto reflection = {pareto_reflecting_function(gpr(res))}")

            if np.all(pareto_reflecting_function(gpr(res)) >= pareto_reflecting_function(
                    blackbox_function.y[0])):
                print("\nNo Pareto point was found. Algorithmic search stopped.")
                break

            if np.any(np.linalg.norm(gpr(res) - np.array(
                    [gpr(x) for x in blackbox_function.x]), axis=1) <= self._min_distance_to_evaluated_points):
                print("\nFound Pareto point is too close to some already evaluated point.")
                break

            print("Evaluating blackbox function...")
            blackbox_function(res)
            print("finished!")
            print(f"Value of blackbox function: ", blackbox_function.y[-1])
            print("Difference to estimation: ", gpr(res)-blackbox_function.y[-1])

            iteration_step += 1

            stop = stopping_criteria(blackbox_function)
