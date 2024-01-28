from functional_tests.blackbox_functions.two_dimensional.zdt1 import ZDT1
from paref.express.express_search import ExpressSearch
from paref.express.info import Info

bbf = ZDT1(input_dimensions=5)

moo = ExpressSearch(bbf)

moo.minimal_search(2)

info = Info(blackbox_function=bbf)

info.minima
info.model_fitness
info.suggestion_pareto_points
info.topology
