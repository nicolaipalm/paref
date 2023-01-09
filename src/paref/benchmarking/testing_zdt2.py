import numpy as np
from scipy.stats import qmc

from paref.express.interfaces.moo_express import MOOExpress
from paref.function_library.zdt2 import ZDT2
from paref.moos.helper_functions.return_pareto_front import return_pareto_front
import plotly.graph_objects as go
from pymoo.indicators.hv import Hypervolume


class TestingZDT2:
    def __init__(self,
                 input_dimensions: int = 2,
                 max_iter_minimizer: int = 100,
                 lh_evaluations: int = 20,
                 reference_point: np.ndarray = np.array([2, 2])):
        self.input_dimensions = input_dimensions
        self.lower_bounds_x = np.zeros(input_dimensions)
        self.upper_bounds_x = np.ones(input_dimensions)
        self.max_iter_minimizer = max_iter_minimizer
        self.lh_evaluations = lh_evaluations

        self.function = ZDT2(input_dimensions=self.input_dimensions)

        # LH evaluation
        [self.function(x) for x in qmc.scale(
            qmc.LatinHypercube(d=len(self.lower_bounds_x)).random(n=lh_evaluations),
            self.lower_bounds_x,
            self.upper_bounds_x,
        )]

        self.real_PF = self.function.return_pareto_front()
        self.hypervolume_max = self.function.calculate_hypervolume_of_pareto_front(reference_point=reference_point)
        self.metric = Hypervolume(ref_point=reference_point, normalize=False)

    def __call__(self, moo: MOOExpress):
        y_initial = self.function.y
        moo(blackbox_function=self.function)

        PF = return_pareto_front([point[1] for point in self.function.evaluations])
        hypervolume_weight = self.metric.do(PF)

        print("HV found/HV max\n:", hypervolume_weight / self.hypervolume_max)

        y = np.array([evaluation[1] for evaluation in self.function.evaluations])

        data = [
            go.Scatter(x=self.real_PF.T[0], y=self.real_PF.T[1], name="Real Pareto front"),
            go.Scatter(x=y[self.lh_evaluations:].T[0], y=y[self.lh_evaluations:].T[1], mode="markers", name="Evaluations"),
            go.Scatter(x=y[:self.lh_evaluations].T[0], y=y[:self.lh_evaluations].T[1], mode="markers",
                       name="Initial Evaluations"),
            go.Scatter(x=PF.T[0], y=PF.T[1], mode="markers", marker=dict(
                color="red", size=8), name="Found Pareto front"),
        ]

        fig1 = go.Figure(data=data)

        fig1.update_layout(
            width=800,
            height=600,
            plot_bgcolor="rgba(0,0,0,0)",
            title=f"zdt2 - {moo.name}: {self.input_dimensions}-dim with rel. HV: {hypervolume_weight / self.hypervolume_max * 100}%",
        )

        fig1.show()

        return self.function
