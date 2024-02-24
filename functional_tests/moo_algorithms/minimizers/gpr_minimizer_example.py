# stopping criteria of MOO algorithm given by maximum iterations
import numpy as np

from functional_tests.scripts.testing_minimizers import TestingMinimizers
from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer

bench = TestingMinimizers()

# Apply MOO
moo = GPRMinimizer()
bench(moo)

print('Value of GPR at true minimum: ', moo._gpr(np.zeros(2)))

moo._gpr.plot_loss()
