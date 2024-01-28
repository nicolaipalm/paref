import numpy as np
from functional_tests.scripts.testing_one_dimensional_sequences import TestingOneDimensionalSequences
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflection_sequences.multi_dimensional.find_1_pareto_points_for_all_components_sequence import \
    Find1ParetoPointsForAllComponentsSequence

epsilon = 1e-2
reference_point = 3 * np.ones(2)
nadir = 10 * np.ones(2)
utopia_point = np.zeros(2)
input_dimensions = 2

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=4)

bench = TestingOneDimensionalSequences(input_dimensions=input_dimensions,
                                       stopping_criteria=stopping_criteria
                                       )

# Apply MOO
sequence = Find1ParetoPointsForAllComponentsSequence()
bench(sequence)
