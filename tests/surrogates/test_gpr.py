import unittest

import numpy as np

from src.surrogates.gpr import GPR


class TestGPR(unittest.TestCase):

    def test_training_is_successful(self):
        gpr = GPR()

        train_x = np.arange(1, 5)
        train_y = np.arange(1, 5).reshape(4, 1)

        self.assertEqual(True, gpr.train(train_x=train_x, train_y=train_y))

    def test_training_with_attention_is_successful(self):
        gpr = GPR()

        train_x = np.arange(1, 5)
        train_y = np.arange(1, 5).reshape(4, 1)

        self.assertEqual(True, gpr.train_with_attention(train_x=train_x, train_y=train_y))

    def test_prediction_is_reasonable(self):
        gpr = GPR()

        train_x = np.arange(1, 5)
        train_y = np.arange(1, 5).reshape(4, 1)

        gpr.train(train_x=train_x, train_y=train_y)

        self.assertAlmostEqual(1.5, gpr(x=np.array([[1.5]]))[0][0], delta=0.1)

    def test_prediction_of_std_is_reasonable(self):
        gpr = GPR()

        train_x = np.arange(1, 5)
        train_y = np.arange(1, 5).reshape(4, 1)

        gpr.train(train_x=train_x, train_y=train_y)

        self.assertAlmostEqual(0, gpr.std(x=np.array([[1.5]]))[0][0], delta=0.1)


if __name__ == '__main__':
    unittest.main()
