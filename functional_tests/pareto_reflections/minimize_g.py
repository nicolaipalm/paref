from functional_tests.scripts.testing_one_dimensional_sequences import TestingOneDimensionalSequences
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflections.minimize_g import MinGParetoReflection

# Meta parameters

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=1)

bench = TestingOneDimensionalSequences(input_dimensions=5,
                                       test_function='dtlz2',
                                       stopping_criteria=stopping_criteria
                                       )


# Apply MOO
class Test(MinGParetoReflection):
    @property
    def g(self):
        return lambda x: x[2]


sequence = Test(blackbox_function=bench.function)
bench(sequence)
