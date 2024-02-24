import numpy as np
from functional_tests.scripts.testing_one_dimensional_sequences import TestingOneDimensionalSequences
from paref.moo_algorithms.minimizer.differential_evolution_minimizer import DifferentialEvolutionMinimizer
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflection_sequences.multi_dimensional.find_edge_points_sequence import FindEdgePointsSequence
from paref.pareto_reflections.priority_search import PrioritySearch

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=1)

bench = TestingOneDimensionalSequences(input_dimensions=5,
                                       test_function='dtlz2',
                                       stopping_criteria=stopping_criteria
                                       )

# find edge points first
stopping_criteria = MaxIterationsReached(max_iterations=3)
sequence = FindEdgePointsSequence()
moo = DifferentialEvolutionMinimizer()
moo.apply_to_sequence(blackbox_function=bench.function,
                      sequence_pareto_reflections=sequence,
                      stopping_criteria=stopping_criteria)

# Apply MOO
sequence = PrioritySearch(blackbox_function=bench.function, priority=np.array([0.1, 0.9, 0.1]), )
bench(sequence)
