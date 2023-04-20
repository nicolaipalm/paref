import numpy as np
from examples.function_library.testing_zdt1 import TestingZDT1
from paref.moo_algorithms.weighted_norm_to_utopia_gpr import WeightedNormToUtopiaGPR

input_dimensions = 5

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 5
max_iter_minimizer = 100
lh_evaluations = 30
epsilon = 1e-1

reference_point = 3 * np.ones(2)

bench = TestingZDT1(input_dimensions=input_dimensions,
                    max_iter_minimizer=max_iter_minimizer,
                    lh_evaluations=lh_evaluations)

moo = WeightedNormToUtopiaGPR(upper_bounds_x=upper_bounds_x,
                              lower_bounds_x=lower_bounds_x,
                              max_evaluations_moo=max_evaluations,
                              epsilon=epsilon,
                              )

bench(moo)
