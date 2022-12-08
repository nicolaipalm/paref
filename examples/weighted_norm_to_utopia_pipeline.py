import numpy as np
from paref.benchmarking.testing_dtlz2 import TestingDTLZ2
from paref.express.weighted_norm_to_utopia_gpr import WeightedNormToUtopiaGPR

input_dimensions = 5

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 10
max_iter_minimizer = 100
lh_evaluations = 20
epsilon = 1e-6

reference_point = 3 * np.ones(2)

bench = TestingDTLZ2(input_dimensions=input_dimensions,
                     max_iter_minimizer=max_iter_minimizer,
                     lh_evaluations=lh_evaluations,
                     )

moo = WeightedNormToUtopiaGPR(upper_bounds_x=upper_bounds_x,
                              lower_bounds_x=lower_bounds_x,
                              max_evaluations_moo=max_evaluations,
                              epsilon=epsilon,
                              potency=np.array([1,1])
                              )

bench(moo)
