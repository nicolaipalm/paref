import numpy as np
from weimoo.function_library.dtlz2 import dtlz2
from weimoo.express.avoiding_with_weighted_norm_to_utopia import AvoidingWithWeightedNormToUtopia

input_dimensions = 20

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 20
max_iter_minimizer = 100
lh_evaluations = 40
epsilon = 2e-1

reference_point = 3 * np.ones(2)

bench = dtlz2(input_dimensions=input_dimensions,
              max_iter_minimizer=max_iter_minimizer,
              output_dimensions=2,
              lh_evaluations=lh_evaluations)

moo = AvoidingWithWeightedNormToUtopia(upper_bounds_x=upper_bounds_x,
                                       lower_bounds_x=lower_bounds_x,
                                       max_evaluations_moo=max_evaluations,
                                       epsilon=epsilon,
                                       nadir=np.array([10, 10]),
                                       )

bench(moo)
