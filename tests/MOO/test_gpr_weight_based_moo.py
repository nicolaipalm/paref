import unittest

import numpy as np

from src.weimoo.moos.gpr_weight_based_moo import GPRWeightBasedMOO
from src.weimoo.interfaces import Function
from src.weimoo.minimizers import DifferentialEvolution
from src.weimoo.weight_functions import ScalarPotency


class TestGPRWeightBasedMOO(unittest.TestCase):
    def test_moo_returns_correct_pareto_point(self):
        weight_function = ScalarPotency(scalar=np.ones(2), potency=1 * np.ones(2))
        moo = GPRWeightBasedMOO(weight_function)

        minimizer = DifferentialEvolution()
        lower_bounds = 10 * np.ones(2)
        upper_bounds = 20 * np.ones(2)

        minimum = 15

        class TestFunction(Function):
            def __call__(self, x):
                return np.array([(x[0] - minimum) ** 2, (x[1] - minimum) ** 2])

        function = TestFunction()
        result = moo(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            number_designs_LH=20,
            max_evaluations=25,
            minimizer=minimizer,
            function=function,
            max_iter_minimizer=1000,
        )

        print(result)

        self.assertEqual(0, np.round(np.sum(minimum - result)))

    if __name__ == "__main__":
        unittest.main()
