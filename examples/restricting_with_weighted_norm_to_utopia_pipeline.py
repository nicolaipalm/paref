import numpy as np

from paref.benchmarking.testing_zdt1 import TestingZDT1
from paref.benchmarking.testing_zdt2 import TestingZDT2
from paref.express.restricting_with_weighted_norm_to_utopia import RestrictingWithWeightedNormToUtopia
from paref.benchmarking.testing_dtlz2 import TestingDTLZ2

input_dimensions = 5

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 5
max_iter_minimizer = 100
lh_evaluations = 35

restricting_point = np.array([0.7, 9])

reference_point = 3 * np.ones(2)
epsilon = 1e-3

bench = TestingZDT1(input_dimensions=input_dimensions,
                    max_iter_minimizer=max_iter_minimizer,
                    lh_evaluations=lh_evaluations)

moo = RestrictingWithWeightedNormToUtopia(upper_bounds_x=upper_bounds_x,
                                          lower_bounds_x=lower_bounds_x,
                                          max_evaluations_moo=max_evaluations,
                                          restricting_point=restricting_point,
                                          nadir=np.array([10, 10]),
                                          potency=np.array([1, 1]),
                                          scalar=np.array([0.1, 1])
                                          )

bench(moo)
