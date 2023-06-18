import numpy as np
from paref.moo_algorithms.multi_dimensional.OUTDATED_find_pareto_point_closest_to_utopia import FindParetoPointClosestToUtopia

from examples.function_library.testing_dtlz2 import TestingDTLZ2

input_dimensions = 20
output_dimensions = 2

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 20
max_iter_minimizer = 100
lh_evaluations = 30
epsilon = 1e-2

reference_point = 3 * np.ones(output_dimensions)

bench = TestingDTLZ2(input_dimensions=input_dimensions,
                     max_iter_minimizer=max_iter_minimizer,
                     lh_evaluations=lh_evaluations,
                     )

moo = FindParetoPointClosestToUtopia(upper_bounds_x=upper_bounds_x,
                                     lower_bounds_x=lower_bounds_x,
                                     max_evaluations_moo=max_evaluations,
                                     epsilon=epsilon,
                                     )

bench(moo)
