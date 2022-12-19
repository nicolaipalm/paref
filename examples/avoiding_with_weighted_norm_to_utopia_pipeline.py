import numpy as np
from paref.benchmarking.testing_dtlz2 import TestingDTLZ2
from paref.benchmarking.testing_zdt2 import TestingZDT2
from paref.benchmarking.testing_zdt1 import TestingZDT1
from paref.express.avoiding_with_weighted_norm_to_utopia import AvoidingWithWeightedNormToUtopia

input_dimensions = 5

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 1
max_iter_minimizer = 100
lh_evaluations = 35
epsilon = 7e-2

reference_point = 3 * np.ones(2)

bench = TestingZDT1(input_dimensions=input_dimensions,
                    max_iter_minimizer=max_iter_minimizer,
                    lh_evaluations=lh_evaluations)

moo = AvoidingWithWeightedNormToUtopia(upper_bounds_x=upper_bounds_x,
                                       lower_bounds_x=lower_bounds_x,
                                       max_evaluations_moo=max_evaluations,
                                       epsilon=epsilon,
                                       nadir=np.array([10, 10]),
                                       scalar=np.array([0.1, 1]),
                                       potency=np.ones(2)
                                       )

bench(moo)
