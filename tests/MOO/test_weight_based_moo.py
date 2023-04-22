import unittest

import numpy as np

from paref import WeightBasedMOO
from paref.interfaces import Function
from paref.optimizers import DifferentialEvolution
from paref import ScalarPotency


class TestWeightBasedMOO(unittest.TestCase):
    def test_moo_returns_correct_pareto_point(self):
        weight_function = ScalarPotency(scalar=2 * np.ones(2), potency=3 * np.ones(2))
        moo = WeightBasedMOO(weight_function)

        minimizer = DifferentialEvolution()
        lower_bounds = np.zeros(2)
        upper_bounds = 100 * np.ones(2)

        minimum = 23

        class TestFunction(Function):
            def __call__(self, x):
                return np.array([(x[0] - minimum) ** 2, (x[1] - minimum) ** 2])

        function = TestFunction()

        self.assertEqual(
            0,
            np.round(
                np.sum(
                    minimum * np.ones(2)
                    - moo(
                        lower_bounds=lower_bounds,
                        upper_bounds=upper_bounds,
                        minimizer=minimizer,
                        function=function,
                        max_evaluations=20,
                    )
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
