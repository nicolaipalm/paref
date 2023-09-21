# stopping criteria of MOO algorithm given by maximum iterations
from examples.scripts.testing_minimizers import TestingMinimizers
from paref.moo_algorithms.minimizer.bayesian_expected_improvement_minimizer import BayesianExpectedImprovementMinimizer

bench = TestingMinimizers()

# Apply MOO
moo = BayesianExpectedImprovementMinimizer()
bench(moo)
