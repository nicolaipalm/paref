import numpy as np
from examples.scripts.testing_one_dimensional_sequences import TestingOneDimensionalSequences
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflections.minimize_weighted_norm_to_utopia import MinimizeWeightedNormToUtopia

# Meta parameters
epsilon = 1e-2
reference_point = 3 * np.ones(2)
nadir = 10 * np.ones(2)
utopia_point = np.zeros(2)

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=2)

bench = TestingOneDimensionalSequences(input_dimensions=5,
                                       stopping_criteria=stopping_criteria
                                       )

# Apply MOO
sequence = MinimizeWeightedNormToUtopia(utopia_point=utopia_point,
                                        potency=1,
                                        scalar=np.array([1,1]))
bench(sequence)
