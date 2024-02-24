from functional_tests.scripts.testing_one_dimensional_sequences import TestingOneDimensionalSequences
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflection_sequences.multi_dimensional.find_1_pareto_points_for_all_components_sequence import \
    Find1ParetoPointsForAllComponentsSequence

input_dimensions = 5

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=4)

bench = TestingOneDimensionalSequences(input_dimensions=input_dimensions,
                                       test_function='dtlz2',
                                       stopping_criteria=stopping_criteria
                                       )

# Apply MOO
sequence = Find1ParetoPointsForAllComponentsSequence()
bench(sequence)
