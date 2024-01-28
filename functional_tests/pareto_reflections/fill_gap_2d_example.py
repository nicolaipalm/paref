import numpy as np
from functional_tests.scripts.testing_one_dimensional_sequences import TestingOneDimensionalSequences
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached

# Meta parameters
from paref.pareto_reflections.fill_gap_2d import FillGap2D

epsilon = 1e-2
reference_point = 3 * np.ones(2)
nadir = 10 * np.ones(2)
utopia_point = np.zeros(2)

point_1 = np.array([0, 1])
point_2 = np.array([0.63, 0.6])

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=2)

bench = TestingOneDimensionalSequences(input_dimensions=5,
                                       stopping_criteria=stopping_criteria,
                                       )

# Apply MOO
sequence = FillGap2D(point_1=point_1, point_2=point_2, utopia_point=utopia_point)
bench(sequence,
      mark_points=['Gap', np.array([point_1, point_2])],
      )
