import numpy as np

from functional_tests.blackbox_functions.dtlz2 import DTLZ2
import plotly.graph_objects as go

from paref.express.express_search import ExpressSearch

bbf = DTLZ2(input_dimensions=3)

bbf.perform_lhc(30)

moo = ExpressSearch(bbf)

moo.minimal_search(4)
moo.priority_search(np.array([0.6, 0.2, 0.2]), 1)

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
