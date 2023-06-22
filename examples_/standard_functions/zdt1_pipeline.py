import numpy as np
from paref.moo_algorithms.multi_dimensional.OUTDATED_find_pareto_point_closest_to_utopia import FindParetoPointClosestToUtopia

from examples.blackbox_functions import TestingZDT1

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

moo = FindParetoPointClosestToUtopia(upper_bounds_x=upper_bounds_x,
                                     lower_bounds_x=lower_bounds_x,
                                     max_evaluations_moo=max_evaluations,
                                     epsilon=epsilon,
                                     )

bench(moo)
