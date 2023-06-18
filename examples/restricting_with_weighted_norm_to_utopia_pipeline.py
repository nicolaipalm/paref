import numpy as np

from examples.function_library.testing_zdt2 import TestingZDT2
from examples.function_library.zdt2 import ZDT2
from paref.moo_algorithms.multi_dimensional.OUTDATED_find_pareto_point_closest_to_utopia import FindParetoPointClosestToUtopia
from paref.pareto_reflections.restrict_by_point import RestrictByPoint

input_dimensions = 2

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 2
max_iter_minimizer = 100
lh_evaluations = 20
epsilon = 1e-2

reference_point = 3 * np.ones(2)

nadir = 10 * np.ones(2)
restricting_point = np.array([0.1, 10])
pareto_reflection = RestrictByPoint(nadir=nadir,
                                    restricting_point=restricting_point)

function = ZDT2(input_dimensions=input_dimensions)

bench = TestingZDT2(input_dimensions=input_dimensions,
                    max_iter_minimizer=max_iter_minimizer,
                    lh_evaluations=lh_evaluations,
                    pareto_reflection=pareto_reflection,
                    function=function,
                    )

moo = FindParetoPointClosestToUtopia(
                                     max_evaluations_moo=max_evaluations,
                                     max_iter_minimizer=max_iter_minimizer,
                                     epsilon=epsilon,
                                     potency=np.array([2, 2]),
                                     scalar=np.array([1, 1]),
                                     # utopia_point=reference_point,
                                     )

bench(moo)
