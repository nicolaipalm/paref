import numpy as np
from paref.benchmarking.testing_zdt1 import TestingZDT1
from paref.express.avoiding_with_weighted_norm_to_utopia import AvoidingWithWeightedNormToUtopia
from paref.express.weighted_norm_to_utopia_gpr import WeightedNormToUtopiaGPR

input_dimensions = 5

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
max_evaluations = 10
max_iter_minimizer = 100
lh_evaluations = 20

reference_point = 3 * np.ones(2)

# find 1 Pareto point corresponding to second coordinate
bench = TestingZDT1(input_dimensions=input_dimensions,
                    max_iter_minimizer=max_iter_minimizer,
                    lh_evaluations=lh_evaluations)

moo = WeightedNormToUtopiaGPR(upper_bounds_x=upper_bounds_x,
                              lower_bounds_x=lower_bounds_x,
                              max_evaluations_moo=1,
                              epsilon=1e-2,
                              potency=np.array([1, 1]),
                              scalar=np.array([0.1, 1]),
                              # utopia_point=reference_point,
                              )
function = bench(moo)

# find 1 Pareto point corresponding to first coordinate
bench = TestingZDT1(input_dimensions=input_dimensions,
                    max_iter_minimizer=max_iter_minimizer,
                    lh_evaluations=0)

moo = WeightedNormToUtopiaGPR(upper_bounds_x=upper_bounds_x,
                              lower_bounds_x=lower_bounds_x,
                              max_evaluations_moo=1,
                              epsilon=0,
                              potency=np.array([1, 1]),
                              scalar=np.array([1, 0.1]),
                              # utopia_point=reference_point,
                              )

bench.lh_evaluations = lh_evaluations
bench.function = function
function = bench(moo)

# find 2 Pareto point
bench = TestingZDT1(input_dimensions=input_dimensions,
                    max_iter_minimizer=max_iter_minimizer,
                    lh_evaluations=0)

moo = WeightedNormToUtopiaGPR(upper_bounds_x=upper_bounds_x,
                              lower_bounds_x=lower_bounds_x,
                              max_evaluations_moo=1,
                              epsilon=0,
                              potency=np.array([5, 5]),
                              scalar=np.array([1, 1]),
                              # utopia_point=reference_point,
                              )
bench.lh_evaluations = lh_evaluations
bench.function = function
function = bench(moo)


# evenly spread
bench = TestingZDT1(input_dimensions=input_dimensions,
                    max_iter_minimizer=max_iter_minimizer,
                    lh_evaluations=0)

moo = AvoidingWithWeightedNormToUtopia(upper_bounds_x=upper_bounds_x,
                                       lower_bounds_x=lower_bounds_x,
                                       max_evaluations_moo=7,
                                       epsilon=1e-3,
                                       nadir=np.array([10, 10]),
                                       scalar=np.array([0.1, 1]),
                                       potency=1 * np.ones(2),
                                       )

bench.lh_evaluations = lh_evaluations
bench.function = function
function = bench(moo)

print(len(function.y))
