# stopping criteria of MOO algorithm given by maximum iterations
from examples.scripts.testing_minimizers import TestingMinimizers
from paref.moo_algorithms.minimizer.differential_evolution_minimizer import DifferentialEvolutionMinimizer

bench = TestingMinimizers()

# Apply MOO
moo = DifferentialEvolutionMinimizer()
bench(moo)
