import numpy as np
from scipy.stats import qmc

from weimoo.moos.helper_functions.ehvi_2d import ehvi_2d
from weimoo.moos.helper_functions.return_pareto_front_2d import (
    return_pareto_front_2d,
)
from weimoo.interfaces.function import Function
from weimoo.interfaces.minimizer import Minimizer
from weimoo.surrogates.gpr import GPR


class EHVI2dMOO:
    def __call__(
        self,
        function: Function,
        minimizer: Minimizer,
        upper_bounds: np.ndarray,
        lower_bounds: np.ndarray,
        number_designs_LH: int,
        max_evaluations: int,
        reference_point: np.ndarray,
        max_iter_minimizer: 1000,
        training_iter: int = 1000,
        learning_rate=0.01,
    ) -> np.ndarray:
        function.clear_evaluations()
        # Setting up a LH for the seed
        train_x = qmc.scale(
            qmc.LatinHypercube(d=len(lower_bounds)).random(n=number_designs_LH),
            lower_bounds,
            upper_bounds,
        )
        evaluations = np.array([function(x) for x in train_x])
        gpr = GPR(training_iter=training_iter, learning_rate=learning_rate)
        for i in range(max_evaluations - number_designs_LH):
            print(
                f"{i + 1}/{max_evaluations - number_designs_LH} Training of the GPR...\n"
            )
            gpr.train(train_x=train_x, train_y=evaluations)
            print(f"\n finished!\n")
            print(f"Starting minimization...")
            pf = return_pareto_front_2d(evaluations)
            res = minimizer(
                function=lambda x: -ehvi_2d(pf, reference_point, gpr(x), gpr.std(x)),
                max_iter=max_iter_minimizer,
                upper_bounds=upper_bounds,
                lower_bounds=lower_bounds,
            )

            train_x = np.append(train_x, res.reshape(1, train_x.shape[1]), axis=0)
            evaluations = np.append(
                evaluations, function(res).reshape(1, evaluations.shape[1]), axis=0
            )

            print(f"\n finished!")

        return evaluations
