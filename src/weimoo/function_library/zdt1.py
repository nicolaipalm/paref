import numpy as np
from scipy.stats import qmc

from weimoo.express.interfaces.moo_express import MOOExpress
from weimoo.function_library.interfaces.function import Function
from weimoo.moos.helper_functions.return_pareto_front_2d import return_pareto_front_2d

from pymoo.factory import get_problem
import plotly.graph_objects as go
from pymoo.indicators.hv import Hypervolume


class zdt1:
    def __init__(self,
                 input_dimensions: int = 2,
                 max_iter_minimizer: int = 100,
                 lh_evaluations: int = 20):
        self.input_dimensions = input_dimensions
        self.lower_bounds_x = np.zeros(input_dimensions)
        self.upper_bounds_x = np.ones(input_dimensions)
        self.max_iter_minimizer = max_iter_minimizer

        problem = get_problem("zdt1", n_var=input_dimensions)

        class ExampleFunction(Function):
            def __call__(self, x):
                self._evaluations.append([x, problem.evaluate(x)])
                return problem.evaluate(x)

        self.function = ExampleFunction()

        # LH evaluation
        [self.function(x) for x in qmc.scale(
            qmc.LatinHypercube(d=len(self.lower_bounds_x)).random(n=lh_evaluations),
            self.lower_bounds_x,
            self.upper_bounds_x,
        )]

        self.real_PF = problem.pareto_front()
        reference_point = np.array([2, 2])
        self.metric = Hypervolume(ref_point=reference_point, normalize=False)
        self.hypervolume_max = self.metric.do(problem.pareto_front())

    def __call__(self, moo: MOOExpress):
        moo(blackbox_function=self.function)

        PF = return_pareto_front_2d([point[1] for point in self.function.evaluations])
        hypervolume_weight = self.metric.do(PF)

        print("HV found/HV max\n:", hypervolume_weight / self.hypervolume_max)

        y = np.array([evaluation[1] for evaluation in self.function.evaluations])

        data = [
            go.Scatter(x=self.real_PF.T[0], y=self.real_PF.T[1], mode="markers"),
            go.Scatter(x=y.T[0], y=y.T[1], mode="markers"),
            go.Scatter(x=PF.T[0], y=PF.T[1], mode="markers"),
        ]

        fig1 = go.Figure(data=data)

        fig1.update_layout(
            width=800,
            height=600,
            plot_bgcolor="rgba(0,0,0,0)",
            title=f"({self.input_dimensions}-dim) zdt1 test function - relative Hypervolume: {hypervolume_weight / self.hypervolume_max * 100}%",
        )

        fig1.show()
