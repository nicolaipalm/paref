import numpy as np
from examples.scripts.testing_one_dimensional_sequences import TestingOneDimensionalSequences
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached

# Meta parameters
from paref.pareto_reflections.avoid_points import AvoidPoints
from paref.pareto_reflections.find_edge_points import FindEdgePoints
from paref.pareto_reflections.operations.compose_reflections import ComposeReflections

epsilon = 1e-1
reference_point = 3 * np.ones(2)
nadir = 10 * np.ones(2)
utopia_point = np.zeros(2)

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=2)

bench = TestingOneDimensionalSequences(input_dimensions=5,
                                       stopping_criteria=stopping_criteria
                                       )

# Apply MOO
pareto_reflection = AvoidPoints(nadir=nadir,
                                epsilon_avoiding_points=np.array([[1, 0]]),  # avoid [1,0]
                                epsilon=epsilon)
sequence = ComposeReflections(pareto_reflection,
                              FindEdgePoints(dimension_domain=2,
                                             dimension=0))
bench(sequence)
