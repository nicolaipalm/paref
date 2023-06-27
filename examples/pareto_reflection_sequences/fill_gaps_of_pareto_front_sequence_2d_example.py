import numpy as np
from examples.scripts.testing_one_dimensional_sequences import TestingOneDimensionalSequences
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached

# Meta parameters
from paref.pareto_reflection_sequences.multi_dimensional.find_edge_points_sequence import FindEdgePointsSequence
from paref.pareto_reflection_sequences.two_dimensional.fill_gaps_of_pareto_front_sequence_2d import \
    FillGapsOfParetoFrontSequence2D

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
sequence = FindEdgePointsSequence()  # find edges first in order to have a Pareto front
bench(sequence)

stopping_criteria = MaxIterationsReached(max_iterations=7)

bench.stopping_criteria = stopping_criteria

sequence = FillGapsOfParetoFrontSequence2D()
bench(sequence)
PF = bench.function.pareto_front
# Sort Pareto points ascending by first component
PF = PF[PF[:, 0].argsort()]
print('Distance of Pareto points to the next: ', np.linalg.norm(PF[:-1] - PF[1:], axis=1))
