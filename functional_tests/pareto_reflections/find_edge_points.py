from functional_tests.scripts.testing_one_dimensional_sequences import TestingOneDimensionalSequences
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached

# Meta parameters
from paref.pareto_reflections.find_edge_points import FindEdgePoints

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=1)

bench = TestingOneDimensionalSequences(input_dimensions=5,
                                       test_function='dtlz2',
                                       stopping_criteria=stopping_criteria
                                       )

# Apply MOO
sequence = FindEdgePoints(dimension=0, blackbox_function=bench.function)
bench(sequence)
