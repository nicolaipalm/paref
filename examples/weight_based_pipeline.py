import numpy as np
import plotly.graph_objects as go
from pymoo.factory import get_problem

from src.weimoo.moos.helper_functions import return_pareto_front_2d
from src.weimoo.moos.weight_based_moo import WeightBasedMOO
from src.weimoo.interfaces import Function
from src.weimoo.minimizers import DifferentialEvolution
from src.weimoo.weight_functions import ScalarPotency

input_dimensions = 10
output_dimensions = 2

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)

minimizer = DifferentialEvolution()

max_iter = 50

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

# Initialize weight function
weight_function = ScalarPotency(potency=2 * np.ones(output_dimensions), scalar=np.ones(output_dimensions))

MOO = WeightBasedMOO(weight_function=weight_function)

result = MOO(function=function, minimizer=minimizer, upper_bounds=upper_bounds_x, lower_bounds=lower_bounds_x,
             max_evaluations=max_iter)

print(result, function(result), weight_function(function(result)))

real_PF = problem.pareto_front()

PF = return_pareto_front_2d([point[1] for point in function.evaluations])

data = [
    go.Scatter(x=PF.T[0], y=PF.T[1], mode="markers"),
    go.Scatter(x=real_PF.T[0], y=real_PF.T[1], mode="markers"),
    go.Scatter(x=np.array([function(result)[0]]), y=np.array([function(result)[1]]), mode="markers"),
]

fig = go.Figure(data=data)
# fig.show()

from pymoo.indicators.hv import Hypervolume

reference_point = np.array([2, 2])
real_PF = problem.pareto_front()

metric = Hypervolume(ref_point=reference_point, normalize=False)

hypervolume_max = metric.do(problem.pareto_front())
hypervolume_weight = metric.do(PF)

print(hypervolume_weight / hypervolume_max)

y = np.array([evaluation[1] for evaluation in function.evaluations])

data = [
    go.Scatter(x=real_PF.T[0], y=real_PF.T[1], mode="markers"),
    go.Scatter(x=y.T[0], y=y.T[1], mode="markers"),
    go.Scatter(x=PF.T[0], y=PF.T[1], mode="markers"),

]

fig1 = go.Figure(data=data)

fig1.update_layout(
    width=800,
    height=600,
    plot_bgcolor="rgba(0,0,0,0)",
    title=f"Weight based MOO:relative Hypervolume: {hypervolume_weight / hypervolume_max * 100}%"
)

fig1.show()
