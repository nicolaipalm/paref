from functional_tests.scripts.testing_gpr_based_moos import TestingGPRBasedMOOs
from paref.moo_algorithms.multi_dimensional.find_1_pareto_points import Find1ParetoPoints
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=2)

bench = TestingGPRBasedMOOs(input_dimensions=5,
                            test_function='zdt1',
                            lh_evaluations=40,
                            stopping_criteria=stopping_criteria
                            )

# Apply MOO
moo = Find1ParetoPoints()
bench(moo)

moo._gpr.plot_loss()
