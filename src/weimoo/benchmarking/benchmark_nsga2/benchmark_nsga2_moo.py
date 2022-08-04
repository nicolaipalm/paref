from pathlib import Path

import numpy as np
import yaml
from pymoo.factory import get_problem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


# MULTIPROCESSING!
#####################
# DOCUMENTATION
#####################
# This file provides a way to benchmark the gpr weight based moo. The meta-data are passed in the meta_data.yaml file
# the data is stored in the data/ directory
# ONLY change the values in the meta_data.yaml file under changeable

# determine how often the moo should be performed
number_runs_moo = 5

path = "data/"

if not Path(path).is_dir():
    raise ValueError(f"The directory: \n{path}\ndoes not exists")

#####################
# FIXED
# loading meta data
with open("meta_data.yaml") as f:
    meta_data = yaml.load(f, Loader=yaml.FullLoader)

# loading variables from meta data
input_dimensions = meta_data.get("input_dimensions")
output_dimension = int(meta_data.get("output_dimension"))

max_evaluations = int(meta_data.get("max_evaluations_blackbox_function"))

problem_name = meta_data.get("benchmarked_function_name")

for j in range(1, number_runs_moo):
    for i, input_dimension in enumerate(input_dimensions):
        print(f"Dimension {input_dimension} ({i + 1}/{len(input_dimensions)})\n")
        input_dimension = int(input_dimension)

        lower_bounds_x = np.zeros(input_dimension)
        upper_bounds_x = np.ones(input_dimension)

        problem = get_problem(problem_name, n_var=input_dimension)

        algorithm = NSGA2(pop_size=5)

        res = minimize(
            problem,
            algorithm,
            termination=("n_evals", max_evaluations),
            seed=10,
            verbose=True,
        )

        # save data to yaml
        content = {
            "meta_data": meta_data,
            "input_dimension": input_dimension,
            "data": [
                {"x": res.X[i].tolist(), "y": res.F[i].tolist()}
                for i, _ in enumerate(res.X)
            ],
        }

        name = str(input_dimension) + "_input_dimensions"

        with open(f"{path}/{name}_{j}.yaml", "w") as f:
            yaml.dump(content, f)
