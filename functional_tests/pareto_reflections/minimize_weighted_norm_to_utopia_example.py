import numpy as np
from functional_tests.scripts.testing_one_dimensional_sequences import TestingOneDimensionalSequences
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflections.minimize_weighted_norm_to_utopia import MinimizeWeightedNormToUtopia

# Meta parameters
utopia_point = np.zeros(3)

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=2)

bench = TestingOneDimensionalSequences(input_dimensions=5,
                                       test_function='dtlz2',
                                       stopping_criteria=stopping_criteria
                                       )

# Apply MOO
sequence = MinimizeWeightedNormToUtopia(utopia_point=utopia_point,
                                        potency=5,
                                        scalar=np.array([1, 1, 1]))
bench(sequence,
      mark_points=['UtopiaPoint', np.array([utopia_point])])
