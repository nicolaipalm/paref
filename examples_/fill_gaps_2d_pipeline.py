import numpy as np
from examples.blackbox_functions import TestingZDT2
from paref.moo_algorithms.two_dimensional.evenly_scanned_2d import EvenlyScanned2d
from paref.moo_algorithms.two_dimensional.fill_gaps_of_pareto_front_2d import FillGaps2d

input_dimensions = 10

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 5
max_iter_minimizer = 100
lh_evaluations = 30

reference_point = 3 * np.ones(2)

bench = TestingZDT2(input_dimensions=input_dimensions,
                    lh_evaluations=lh_evaluations,
                    )

moo = EvenlyScanned2d(upper_bounds_x=upper_bounds_x,
                      lower_bounds_x=lower_bounds_x,
                      max_evaluations_moo=2,
                      max_iter_minimizer=max_iter_minimizer,
                      min_distance_to_evaluated_points=2e-2,
                      restricting_point_wrt_previous_evaluated_point=True,
                      )

bench(moo)
moo = FillGaps2d(upper_bounds_x=upper_bounds_x,
                 lower_bounds_x=lower_bounds_x,
                 max_evaluations_moo=max_evaluations,
                 max_iter_minimizer=max_iter_minimizer,
                 min_distance_to_evaluated_points=2e-2,
                 )

bench(moo)
