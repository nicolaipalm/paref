import numpy as np
from pymoo.factory import get_problem
from scipy.stats import qmc

from src.interfaces.function import Function
from src.surrogates.gpr import GPR

max_iter = 20
input_dimensions = 2
output_dimensions = 2

number_designs = max_iter

lower_bounds_design = np.zeros(input_dimensions)
upper_bounds_design = np.ones(input_dimensions)
initial_LH = qmc.scale(
    qmc.LatinHypercube(d=len(lower_bounds_design)).random(n=number_designs),
    lower_bounds_design,
    upper_bounds_design,
)
problem = get_problem("dtlz2",
                      n_var=input_dimensions,
                      n_obj=output_dimensions)


class ExampleFunction(Function):
    def __call__(self, x):
        if len(self._evaluations) < max_iter:
            self._evaluations.append([
                x,
                problem.evaluate(x)
            ])
        return problem.evaluate(x)


# Initialiaze the function
function = ExampleFunction()

input_parameters = initial_LH
output_parameters = function(initial_LH)

surrogate = GPR()

print(initial_LH)
print(function(initial_LH))

surrogate.train(train_x=input_parameters, train_y=output_parameters)

print(output_parameters-surrogate(initial_LH))

