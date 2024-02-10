import numpy as np

from functional_tests.blackbox_functions.dtlz2 import DTLZ2
import plotly.graph_objects as go

from functional_tests.blackbox_functions.zdt2 import ZDT2
from paref.express.express_search import ExpressSearch
from paref.moo_algorithms.multi_dimensional.find_edge_points import FindEdgePoints
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached

bbf = ZDT2(input_dimensions=5)

bbf.perform_lhc(30)

moo = ExpressSearch(bbf, )
moo.minimal_search(3)
moo.priority_search(np.array([0.8, 0.2]), 1)

real_PF = bbf.return_true_pareto_front()

data = [
    go.Scatter(x=real_PF.T[0],
               y=real_PF.T[1],
               line=dict(width=4),
               name='Real Pareto front'),
    go.Scatter(x=bbf.y[30:33].T[0],
               y=bbf.y[30:33].T[1],
               mode='markers',
               marker=dict(size=10),
               name='Minimal Search'),
    go.Scatter(x=bbf.y[33:].T[0],
               y=bbf.y[33:].T[1],
               mode='markers',
               marker=dict(size=10),
               name='Priority Search'),
    go.Scatter(x=bbf.y[:30].T[0],
               y=bbf.y[:30].T[1],
               mode='markers',
               name='Initial Evaluations'),
]

fig1 = go.Figure(data=data)

fig1.update_layout(
    width=500,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(
        x=0.1,
        y=0.9, )
)

fig1.show()

input()

######################
# 3 dimensions

bbf = DTLZ2(input_dimensions=5)

bbf.perform_lhc(30)
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
