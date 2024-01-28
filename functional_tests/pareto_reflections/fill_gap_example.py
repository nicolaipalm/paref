import numpy as np
from functional_tests.scripts.testing_one_dimensional_sequences import TestingOneDimensionalSequences
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached

# Meta parameters
from paref.pareto_reflections.fill_gap import FillGap

point_1 = np.array([0, 1])
point_2 = np.array([0.63, 0.6])
gap_points = np.array([point_1, point_2])

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=2)

bench = TestingOneDimensionalSequences(input_dimensions=5,
                                       stopping_criteria=stopping_criteria,
                                       )

# Apply MOO
sequence = FillGap(gap_points=gap_points, dimension_domain=2)
bench(sequence,
      mark_points=['Gap', gap_points],
      )
