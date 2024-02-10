import numpy as np

from functional_tests.blackbox_functions.dtlz2 import DTLZ2
import plotly.graph_objects as go

from paref.moo_algorithms.multi_dimensional.find_edge_points import FindEdgePoints
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached

bbf = DTLZ2(input_dimensions=5)

bbf.perform_lhc(30)
# moo = ExpressSearch(bbf,)
# #moo.search_for_minima(1, 1)
# print(moo.minima_components)
# moo.minimal_search(7)
# #moo.priority_search(np.array([0.1, 0.8, 0.1]), 1)
#
# print('Max points:', moo.max_point,
#       '\nMinima components:', moo.minima_components,
#       '\nPriority:', moo.priority_point, )

moo = FindEdgePoints()
moo(blackbox_function=bbf, stopping_criteria=MaxIterationsReached(max_iterations=7))

pf = np.array(bbf.return_true_pareto_front())

fig = go.Figure(data=[go.Mesh3d(x=pf.T[0],
                                y=pf.T[1],
                                z=pf.T[2],
                                opacity=0.5,
                                color='rgba(244,22,100,0.6)'
                                ),
                      go.Scatter3d(x=bbf.y[30:].T[0], y=bbf.y[30:].T[1], z=bbf.y[30:].T[2],
                                   mode='markers')
                      ])

fig.show()
