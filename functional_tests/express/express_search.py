import numpy as np

from functional_tests.blackbox_functions.two_dimensional.zdt1 import ZDT1
from paref.express.express_search import ExpressSearch

bbf = ZDT1(input_dimensions=2)

moo = ExpressSearch(bbf)

moo.minimal_search(10)

moo.priority_search(np.array([0.2, 0.8]), 4)

moo.minimize_g(lambda x: x[1] + 2 * x[0], 4)

print('Max points:', moo.max_point,
      '\nMinima components:', moo.minima_components,
      '\nMinima g:', moo.min_g)
