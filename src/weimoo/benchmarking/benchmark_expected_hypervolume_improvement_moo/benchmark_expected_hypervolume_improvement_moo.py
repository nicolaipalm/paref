from pathlib import Path

import numpy as np
import yaml
from pymoo.factory import get_problem

from weimoo.interfaces.function import Function
from weimoo.minimizers.differential_evolution import DifferentialEvolution
from weimoo.moos.expected_hypervolume_improvement_2d_moo import EHVI2dMOO
from weimoo.weight_functions.scalar_potency import ScalarPotency

# MULTIPROCESSING!
#####################
# DOCUMENTATION
#####################
# This file provides a way to benchmark the expected hypervolume MOO.
# The meta-data are passed in the meta_data.yaml file
# the data is stored in the data/ directory
# ONLY change the values in the meta_data.yaml file under changeable

# determine how often the moo should be performed
number_runs_moo = 9

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

max_iter_minimizer = int(meta_data.get("max_iter_minimizer"))
max_evaluations = int(meta_data.get("max_evaluations_blackbox_function"))

training_iter = int(meta_data.get("training_iter_gpr"))

potency = np.array(meta_data.get("potency_weights"))
scalar = np.array(meta_data.get("scalar_weights"))

number_designs_LH = int(meta_data.get("number_designs_LH"))

reference_point = np.array(meta_data.get("reference_point"))

problem_name = meta_data.get("benchmarked_function_name")

for j in range(4, number_runs_moo):

    for i, input_dimension in enumerate(input_dimensions):
        print(f"Dimension {input_dimension} ({i + 1}/{len(input_dimensions)})\n")
        input_dimension = int(input_dimension)

        lower_bounds_x = np.zeros(input_dimension)
        upper_bounds_x = np.ones(input_dimension)

        minimizer = DifferentialEvolution()

        problem = get_problem(problem_name, n_var=input_dimension)

        # Initialize weight function
        weight_function = ScalarPotency(potency=potency, scalar=scalar)

        ####################
        class ExampleFunction(Function):
            def __call__(self, x):
                self._evaluations.append([x, problem.evaluate(x)])
                return problem.evaluate(x)

        # Initialize the function
        function = ExampleFunction()

        MOO = EHVI2dMOO()

        result = MOO(
            function=function,
            minimizer=minimizer,
            upper_bounds=upper_bounds_x,
            lower_bounds=lower_bounds_x,
            number_designs_LH=number_designs_LH,
            max_evaluations=max_evaluations,
            reference_point=reference_point,
            max_iter_minimizer=max_iter_minimizer,
            training_iter=training_iter,
        )

        # save data to yaml
        content = {
            "meta_data": meta_data,
            "input_dimension": input_dimension,
            "data": [
                {"x": evaluation[0].tolist(), "y": evaluation[1].tolist()}
                for evaluation in function.evaluations
            ],
        }

        name = str(input_dimension) + "_input_dimensions"

        with open(f"{path}/{name}_{j}.yaml", "w") as f:
            yaml.dump(content, f)
