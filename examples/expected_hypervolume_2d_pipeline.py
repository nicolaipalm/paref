import numpy as np

from paref.benchmarking.testing_dtlz2 import TestingDTLZ2
from paref.express.expected_hypervolume_improvement_2d import ExpectedHypervolumeImprovement2d

input_dimensions = 5
output_dimensions = 2

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 10
max_iter_minimizer = 100
lh_evaluations = 20
epsilon = 1e-6

reference_point = 3 * np.ones(output_dimensions)

bench = TestingDTLZ2(input_dimensions=input_dimensions,
                     max_iter_minimizer=max_iter_minimizer,
                     lh_evaluations=lh_evaluations,
                     )

moo = ExpectedHypervolumeImprovement2d(upper_bounds_x=upper_bounds_x,
                                       lower_bounds_x=lower_bounds_x,
                                       reference_point=reference_point,
                                       max_evaluations_moo=max_evaluations,
                                       )

bench(moo)
