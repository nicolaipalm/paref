import numpy as np

from paref.benchmarking.testing_zdt2 import TestingZDT2
from paref.moo_algorithms.weighted_norm_to_utopia_gpr import WeightedNormToUtopiaGPR

input_dimensions = 2

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 1
max_iter_minimizer = 1000
lh_evaluations = 5
epsilon = 1e-2

reference_point = 3 * np.ones(2)

bench = TestingZDT2(input_dimensions=input_dimensions,
                    max_iter_minimizer=max_iter_minimizer,
                    lh_evaluations=lh_evaluations,
                    )

moo = WeightedNormToUtopiaGPR(upper_bounds_x=upper_bounds_x,
                              lower_bounds_x=lower_bounds_x,
                              max_evaluations_moo=max_evaluations,
                              max_iter_minimizer=max_iter_minimizer,
                              epsilon=epsilon,
                              potency=np.array([2, 2]),
                              scalar=np.array([1, 1]),
                              # utopia_point=reference_point,
                              )

bench(moo)
