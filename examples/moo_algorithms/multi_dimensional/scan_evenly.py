from examples.scripts.testing_gpr_based_moos import TestingGPRBasedMOOs
from paref.moo_algorithms.multi_dimensional.find_edge_points import FindEdgePoints
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.moo_algorithms.multi_dimensional.fill_gaps_of_pareto_front import FillGapsOfParetoFront

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
moo = FillGapsOfParetoFront()
bench(moo)
