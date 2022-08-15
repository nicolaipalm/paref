from pathlib import Path

import numpy as np
import yaml
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

#####################
# DOCUMENTATION
#####################
# This file provides a way to benchmark the gpr weight based moo.
# The meta-data are passed in the meta_data.yaml file
# the data is stored in the data/ directory
# ONLY change the values in the meta_data.yaml file under changeable

# determine how often the moo should be performed
number_runs_moo = 2

path = 'data/zdt1'

if not Path(path).is_dir():
    raise ValueError(f'The directory: \n{path}\ndoes not exists')

#####################
# FIXED
# loading meta data
with open('meta_data.yaml') as f:
    meta_data = yaml.load(f, Loader=yaml.FullLoader)

# loading variables from meta data
input_dimensions = meta_data.get('input_dimensions')
output_dimension = int(meta_data.get('output_dimension'))

max_evaluations = int(meta_data.get('max_evaluations_blackbox_function'))

problem_name = meta_data.get('benchmarked_function_name')

for j in range(1, number_runs_moo):
    for i, input_dimension in enumerate(input_dimensions):
        print(f'Dimension {input_dimension} ({i + 1}/{len(input_dimensions)})\n')
        input_dimension = int(input_dimension)

        lower_bounds_x = np.zeros(input_dimension)
        upper_bounds_x = np.ones(input_dimension)

        # problem = get_problem(problem_name, n_var=input_dimension)

        from pymoo.core.problem import Problem

        input_evaluations_raw = []
        output_evaluations_raw = []

        class ZDT1(Problem):
            def __init__(self, n_var=input_dimension, **kwargs):
                super().__init__(n_var=n_var,
                                 n_obj=2,
                                 n_ieq_constr=0,
                                 xl=0,
                                 xu=1,
                                 vtype=float)

            def _calc_pareto_front(self, n_pareto_points=100):
                x = np.linspace(0, 1, n_pareto_points)
                return np.array([x, 1 - np.sqrt(x)]).T

            def _evaluate(self, x, out, *args, **kwargs):
                f1 = x[:, 0]
                g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis=1)
                f2 = g * (1 - np.power((f1 / g), 0.5))
                output_evaluations_raw.append([f1, f2])
                input_evaluations_raw.append(x)
                out['F'] = np.column_stack([f1, f2])

        problem = ZDT1(input_dimension)

        population_size = 20
        algorithm = NSGA2(population_size)

        res = minimize(
            problem,
            algorithm,
            termination=('n_gen', max_evaluations / population_size),
            # seed=1,
            verbose=False,
            save_history=True,
        )

        evaluations = []
        for array1 in input_evaluations_raw:
            for array2 in np.array(array1):
                evaluations.append(array2)

        input_evaluations = np.array(evaluations)

        evaluations = []
        for array1 in output_evaluations_raw:
            for array2 in np.array(array1).T:
                evaluations.append(array2)

        output_evaluations = np.array(evaluations)

        # save data to yaml
        content = {
            'meta_data': meta_data,
            'input_dimension': input_dimension,
            'data': [
                # {"x": res.X[i].tolist(), "y": res.F[i].tolist()}
                {'x': input_evaluations[i].tolist(), 'y': output_evaluations[i].tolist()}
                for i, _ in enumerate(input_evaluations)
            ],
        }

        name = str(input_dimension) + '_input_dimensions'

        with open(f'{path}/{name}_{j}.yaml', 'w') as f:
            yaml.dump(content, f)

        x = pd.DataFrame(content['data'])
        x.to_csv('data/zdt1/csv/'+f'{name}_{j}.csv')
