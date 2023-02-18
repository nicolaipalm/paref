import numpy as np
from paref.benchmarking.testing_zdt2 import TestingZDT2
from paref.express.evenly_scanned_2d import EvenlyScanned2d

input_dimensions = 20

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 5
max_iter_minimizer = 1000
lh_evaluations = 30

reference_point = 3 * np.ones(2)

bench = TestingZDT2(input_dimensions=input_dimensions,
                    max_iter_minimizer=max_iter_minimizer,
                    lh_evaluations=lh_evaluations,
                    )

moo = EvenlyScanned2d(upper_bounds_x=upper_bounds_x,
                      lower_bounds_x=lower_bounds_x,
                      max_evaluations_moo=max_evaluations,
                      restricting_point_wrt_previous_evaluated_point=True,
                      )

bench(moo)
