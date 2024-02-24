import numpy as np
from functional_tests.scripts.testing_one_dimensional_sequences import TestingOneDimensionalSequences
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached

# Meta parameters
from paref.pareto_reflections.fill_gap import FillGap

point_1 = np.array([0, 0, 1])
point_2 = np.array([1, 0, 0])
point_3 = np.array([0, 1, 0])
gap_points = np.array([point_1, point_2, point_3])

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=2)

bench = TestingOneDimensionalSequences(input_dimensions=5,
                                       test_function='dtlz2',
                                       stopping_criteria=stopping_criteria,
                                       )

# Apply MOO
sequence = FillGap(gap_points=gap_points, )
bench(sequence,
      mark_points=['Gap', gap_points],
      )
