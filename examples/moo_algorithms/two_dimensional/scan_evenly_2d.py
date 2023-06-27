import numpy as np
from examples.scripts.testing_gpr_based_moos import TestingGPRBasedMOOs
from paref.moo_algorithms.multi_dimensional.find_edge_points import FindEdgePoints
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached

# Meta parameters
from paref.moo_algorithms.two_dimensional.fill_gaps_of_pareto_front_2d import FillGapsOfParetoFront2D

epsilon = 1e-2
reference_point = 3 * np.ones(2)
nadir = 10 * np.ones(2)
utopia_point = np.zeros(2)

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=5)

bench = TestingGPRBasedMOOs(input_dimensions=5,
                            max_iter_minimizer=100,
                            lh_evaluations=40,
                            stopping_criteria=stopping_criteria
                            )

# Apply MOO
sequence = FindEdgePoints()  # search for edge points first
bench(sequence)


bench.stopping_criteria = MaxIterationsReached(max_iterations=3)
# Apply MOO
moo = FillGapsOfParetoFront2D()
bench(moo)
