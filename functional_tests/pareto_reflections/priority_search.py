import numpy as np
from functional_tests.scripts.testing_one_dimensional_sequences import TestingOneDimensionalSequences
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflections.priority_search import PrioritySearch

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

bench.function(np.array([1, 0, 0, 0, 0]))
bench.function(np.zeros(5))
# Apply MOO
sequence = PrioritySearch(blackbox_function=bench.function, priority=[0.1, 0.9], )
bench(sequence)
