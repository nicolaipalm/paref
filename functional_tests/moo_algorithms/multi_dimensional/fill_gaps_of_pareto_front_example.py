import numpy as np
from functional_tests.scripts.testing_gpr_based_moos import TestingGPRBasedMOOs
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.moo_algorithms.two_dimensional.fill_gaps_of_pareto_front_2d import FillGapsOfParetoFront2D

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=3)

input_dimensions = 5
bench = TestingGPRBasedMOOs(input_dimensions=input_dimensions,
                            test_function='dtlz2',
                            lh_evaluations=20,
                            stopping_criteria=stopping_criteria
                            )

pareto_point_1 = np.zeros(input_dimensions)
bench.function(pareto_point_1)

pareto_point_2 = np.zeros(input_dimensions)
pareto_point_2[0] = 0.5
bench.function(pareto_point_2)

# Apply MOO
moo = FillGapsOfParetoFront2D()
bench(moo,
      mark_points=['Gap of Pareto points', np.array([bench.function(pareto_point_1), bench.function(pareto_point_2)])])
