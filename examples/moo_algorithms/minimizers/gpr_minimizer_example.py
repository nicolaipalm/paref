# stopping criteria of MOO algorithm given by maximum iterations
from examples.scripts.testing_minimizers import TestingMinimizers
from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer

bench = TestingMinimizers()

# Apply MOO
moo = GPRMinimizer()
bench(moo)
