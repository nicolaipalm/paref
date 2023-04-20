import numpy as np

from paref.benchmarking.testing_zdt1 import TestingZDT1
from paref.moo_algorithms.expected_hypervolume_improvement_2d import ExpectedHypervolumeImprovement2d

input_dimensions = 2
output_dimensions = 2

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 5
max_iter_minimizer = 100
lh_evaluations = 5
epsilon = 1e-6

reference_point = np.array([3, 3])

bench = TestingZDT1(input_dimensions=input_dimensions,
                    max_iter_minimizer=max_iter_minimizer,
                    lh_evaluations=lh_evaluations,
                    )

moo = ExpectedHypervolumeImprovement2d(upper_bounds_x=upper_bounds_x,
                                       lower_bounds_x=lower_bounds_x,
                                       reference_point=reference_point,
                                       max_evaluations_moo=max_evaluations,
                                       )

bench(moo)
