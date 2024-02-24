from functional_tests.blackbox_functions.dtlz2 import DTLZ2
from functional_tests.blackbox_functions.zdt1 import ZDT1
from paref.express.express_search import ExpressSearch
from paref.express.info import Info

bbf = DTLZ2(input_dimensions=3)
bbf = ZDT1(input_dimensions=2)
bbf.perform_lhc(30)

moo = ExpressSearch(bbf)

moo.minimal_search(3)

info = Info(blackbox_function=bbf)

info.minima
info.model_fitness
info.suggestion_pareto_points
info.topology

print(info.edge_points)
