import numpy as np

from functional_tests.blackbox_functions.two_dimensional.zdt1 import ZDT1
from paref.express.express_search import ExpressSearch

bbf = ZDT1(input_dimensions=2)

bbf.perform_lhc(20)
moo = ExpressSearch(bbf, constraints=np.array([0.5, 10]))
moo.search_for_minima(2, 1)
print(moo.minima_components)
moo.minimal_search(5)
moo.priority_search(np.array([0.2, 0.8]), 4)

print('Max points:', moo.max_point,
      '\nMinima components:', moo.minima_components,
      '\nPriority:', moo.priority_point, )
