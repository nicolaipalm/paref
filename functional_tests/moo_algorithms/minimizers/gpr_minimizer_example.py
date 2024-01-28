# stopping criteria of MOO algorithm given by maximum iterations

from functional_tests.scripts.testing_minimizers import TestingMinimizers
from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer

bench = TestingMinimizers()

# Apply MOO
moo = GPRMinimizer(preprocess=False)
bench(moo)

moo._gpr.plot_loss()
