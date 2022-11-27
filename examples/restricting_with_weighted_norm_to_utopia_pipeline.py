import numpy as np

from weimoo.express.restricting_with_weighted_norm_to_utopia import RestrictingWithWeightedNormToUtopia
from weimoo.function_library.dtlz2 import dtlz2

input_dimensions = 5

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 10
max_iter_minimizer = 100
lh_evaluations = 20

restricting_point = np.array([0.9, 0.9])

reference_point = 3 * np.ones(2)

bench = dtlz2(input_dimensions=input_dimensions,
              max_iter_minimizer=max_iter_minimizer,
              output_dimensions=2,
              lh_evaluations=lh_evaluations)

moo = RestrictingWithWeightedNormToUtopia(upper_bounds_x=upper_bounds_x,
                                          lower_bounds_x=lower_bounds_x,
                                          max_evaluations_moo=max_evaluations,
                                          restricting_point=restricting_point,
                                          nadir=np.array([10, 10]),
                                          potency=np.array([1, 1]),
                                          )

bench(moo)
