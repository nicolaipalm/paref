from functional_tests.scripts.testing_gpr_based_moos import TestingGPRBasedMOOs
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.moo_algorithms.multi_dimensional.find_edge_points import FindEdgePoints

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=7)

bench = TestingGPRBasedMOOs(input_dimensions=5,
                            max_iter_minimizer=250,
                            lh_evaluations=30,
                            stopping_criteria=stopping_criteria
                            )

# Apply MOO
moo = FindEdgePoints()
bench(moo)
print(moo.best_fits)
