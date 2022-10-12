from pathlib import Path

from pymoo.factory import get_problem

from weimoo.benchmarking.evaluation.evaluation_moo import EvaluationMOO

moo_names = [  # "gpr_weight_based_moo",
    # "gpr_multiple_weights_based_moo",
    # "expected_hypervolume_improvement_moo",
    'nsga2',
    # "latin_hypercube",
]

problem = get_problem('zdt1', n_var=2)

real_PF = problem.pareto_front()

for moo in moo_names:
    e = EvaluationMOO(
        paths=[
            Path(f'../benchmark_{moo}/data/zdt1/{dimension}_input_dimensions_{j}.yaml')
            for j in range(1, 2)
            for dimension in [2, 5, 15, 25, 30]
        ]
    )
    e.plot_dots_2d_with_pareto_front(pareto_front=real_PF)


f = EvaluationMOO(
    paths=[
        Path(f'../benchmark_{moo}/data/zdt1/{dimension}_input_dimensions_1.yaml')
        for dimension in [2, 5, 15, 25, 30]
        for moo in moo_names
    ]
)
