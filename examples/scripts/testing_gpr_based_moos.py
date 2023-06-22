from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go
from pymoo.indicators.hv import Hypervolume
from scipy.stats import qmc

from examples.blackbox_functions.two_dimensional.zdt1 import ZDT1
from examples.blackbox_functions.two_dimensional.zdt2 import ZDT2
from paref.interfaces.moo_algorithms.paref_moo import ParefMOO
from paref.interfaces.moo_algorithms.stopping_criteria import StoppingCriteria
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections


class TestingGPRBasedMOOs:
    def __init__(self,
                 stopping_criteria: StoppingCriteria,
                 test_function: str = "zdt2",
                 sequence_pareto_reflection: Optional[Union[SequenceParetoReflections, ParetoReflection]] = None,
                 input_dimensions: int = 5,
                 max_iter_minimizer: int = 100,
                 lh_evaluations: int = 20,
                 reference_point: np.ndarray = np.array([2, 2])):
        self.input_dimensions = input_dimensions
        self.max_iter_minimizer = max_iter_minimizer
        self.lh_evaluations = lh_evaluations
        self.sequence = sequence_pareto_reflection
        self.stopping_criteria = stopping_criteria

        test_function_set = ["zdt1", "zdt2"]

        if test_function == "zdt2":
            self.function = ZDT2(input_dimensions=input_dimensions)

        elif test_function == "zdt1":
            self.function = ZDT1(input_dimensions=input_dimensions)

        else:
            raise ValueError(f"Test function must be one of {test_function_set}!")

        # LH evaluation
        [self.function(x) for x in qmc.scale(
            qmc.LatinHypercube(d=self.function.dimension_design_space).random(n=lh_evaluations),
            self.function.design_space.lower_bounds,
            self.function.design_space.upper_bounds,
        )]

        self.real_PF = self.function.return_true_pareto_front()
        self.hypervolume_max = self.function.calculate_hypervolume_of_pareto_front(reference_point=reference_point)
        self.metric = Hypervolume(ref_point=reference_point, normalize=False)

    def __call__(self, moo: ParefMOO):
        if self.sequence is None:
            moo(blackbox_function=self.function,
                stopping_criteria=self.stopping_criteria)
        else:
            moo.apply_to_sequence(blackbox_function=self.function,
                                  sequence_pareto_reflections=self.sequence,
                                  stopping_criteria=self.stopping_criteria)

        PF = self.function.pareto_front
        hypervolume_weight = self.metric.do(PF)

        print('HV found/HV max\n:', hypervolume_weight / self.hypervolume_max)

        y = np.array([evaluation[1] for evaluation in self.function.evaluations])

        data = [
            go.Scatter(x=self.real_PF.T[0], y=self.real_PF.T[1], name='Real Pareto front'),
            go.Scatter(x=y[self.lh_evaluations:].T[0], y=y[self.lh_evaluations:].T[1], mode='markers',
                       name='Evaluations'),
            go.Scatter(x=y[:self.lh_evaluations].T[0], y=y[:self.lh_evaluations].T[1], mode='markers',
                       name='Initial Evaluations'),
            go.Scatter(x=PF.T[0], y=PF.T[1], mode='markers', marker=dict(
                color='red', size=8), name='Found Pareto front'),
        ]

        fig1 = go.Figure(data=data)

        fig1.update_layout(
            width=800,
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            title=f'zdt2: {self.input_dimensions}-dim input with rel. HV: '
                  f'{hypervolume_weight / self.hypervolume_max * 100}%',
        )

        fig1.show()

        return self.function
