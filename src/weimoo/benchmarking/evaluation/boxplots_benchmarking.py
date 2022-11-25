from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from pymoo.factory import get_problem

from weimoo.benchmarking.evaluation.evaluation_moo import EvaluationMOO

moo_names = [
    "gpr_weight_based_moo",
    "gpr_multiple_weights_based_moo",
    "expected_hypervolume_improvement_moo",
    'nsga2',
    "latin_hypercube",
]

problem = get_problem('zdt1', n_var=2)

reference_point = np.array([2, 4])

real_PF = problem.pareto_front()

for moo in moo_names:
    data = []
    for dimension in [2, 5, 15, 25, 30]:
        e = EvaluationMOO(
            paths=[
                Path(
                    f'../benchmark_{moo}/data/zdt1/{dimension}_input_dimensions_{j}.yaml'
                )
                for j in range(1, 5)
            ]
        )

        data.append(
            go.Box(
                y=[
                    evaluation_moo.calculate_relative_hyper_volume_2d(
                        reference_point=reference_point, pareto_front=real_PF
                    )
                    for evaluation_moo in e.evaluations_moos
                ],
                boxpoints='all',  # can also be outliers, or suspectedoutliers, or False
                jitter=0.3,  # add some jitter for a better separation between points
                pointpos=-1.8,  # relative position of points wrt box
                name=f'{dimension}-dimensional input',
            )
        )

    fig1 = go.Figure(data=data, layout_yaxis_range=[0, 1])

    title = f"{e.evaluations_moos[0].meta_data.get('MOO_name')} rel hv with ref point {reference_point.tolist()}"

    fig1.update_layout(
        width=800,
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        title=title,
    )

    fig1.show()
