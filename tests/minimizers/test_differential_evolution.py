import unittest

import numpy as np

from paref.interfaces import Function
from paref.optimizers import DifferentialEvolution


class MyTestCase(unittest.TestCase):
    def test_something(self):
        minimizer = DifferentialEvolution()
        lower_bounds = -100 * np.ones(1)
        upper_bounds = 100 * np.ones(1)

        minimum = 23

        class TestFunction(Function):
            def __call__(self, x):
                return (x - minimum) ** 2

        function = TestFunction()

        self.assertAlmostEqual(
            minimum,
            minimizer(
                function=function, upper_bounds=upper_bounds, lower_bounds=lower_bounds
            ),
            delta=0.1,
        )


if __name__ == "__main__":
    unittest.main()
