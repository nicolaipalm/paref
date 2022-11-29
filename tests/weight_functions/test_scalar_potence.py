import unittest

import numpy as np

from src.paref.pareto_reflecting_library import ScalarPotency


class TestScalarPotency(unittest.TestCase):
    def test_something(self):
        weight_function = ScalarPotency(scalar=2 * np.ones(5), potency=3 * np.ones(5))
        self.assertEqual(
            5 * 2 * 2 ** 3, weight_function(2 * np.ones(5))
        )  # add assertion here


if __name__ == "__main__":
    unittest.main()
