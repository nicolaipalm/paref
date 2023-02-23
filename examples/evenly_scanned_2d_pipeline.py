import numpy as np
from paref.benchmarking.testing_zdt2 import TestingZDT2
from paref.express.evenly_scanned_2d import EvenlyScanned2d

input_dimensions = 2

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 3
max_iter_minimizer = 100
lh_evaluations = 30

reference_point = 3 * np.ones(2)

bench = TestingZDT2(input_dimensions=input_dimensions,
                    lh_evaluations=lh_evaluations,
                    )

moo = EvenlyScanned2d(upper_bounds_x=upper_bounds_x,
                      lower_bounds_x=lower_bounds_x,
                      max_evaluations_moo=max_evaluations,
                      max_iter_minimizer=max_iter_minimizer,
                      min_distance_to_evaluated_points=2e-2,
                      restricting_point_wrt_previous_evaluated_point=True,
                      )

bench(moo)
