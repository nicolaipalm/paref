from typing import Union

import numpy as np
import plotly.graph_objects as go
from pymoo.indicators.hv import Hypervolume

from examples.blackbox_functions.two_dimensional.zdt1 import ZDT1
from examples.blackbox_functions.two_dimensional.zdt2 import ZDT2
from paref.interfaces.moo_algorithms.stopping_criteria import StoppingCriteria
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.moo_algorithms.minimizer.differential_evolution_minimizer import DifferentialEvolutionMinimizer


class TestingOneDimensionalSequences:
    def __init__(self,
                 stopping_criteria: StoppingCriteria,
                 test_function: str = "zdt2",
                 input_dimensions: int = 5,
                 reference_point: np.ndarray = np.array([2, 2])):
        self.input_dimensions = input_dimensions
        self.stopping_criteria = stopping_criteria

        test_function_set = ["zdt1", "zdt2"]

        if test_function == "zdt2":
            self.function = ZDT2(input_dimensions=input_dimensions)

        elif test_function == "zdt1":
            self.function = ZDT1(input_dimensions=input_dimensions)

        else:
            raise ValueError(f"Test function must be one of {test_function_set}!")

        self.real_PF = self.function.return_true_pareto_front()
        self.hypervolume_max = self.function.calculate_hypervolume_of_pareto_front(reference_point=reference_point)
        self.metric = Hypervolume(ref_point=reference_point, normalize=False)

    def __call__(self, sequence: Union[SequenceParetoReflections, ParetoReflection]):
        moo = DifferentialEvolutionMinimizer()
        moo.apply_to_sequence(blackbox_function=self.function,
                              sequence_pareto_reflections=sequence,
                              stopping_criteria=self.stopping_criteria)

        y = np.array([evaluation[1] for evaluation in self.function.evaluations])

        data = [
            go.Scatter(x=self.real_PF.T[0], y=self.real_PF.T[1], name='Real Pareto front'),
            go.Scatter(x=y.T[0], y=y.T[1], mode='markers',
                       name='Evaluations'),
        ]

        fig1 = go.Figure(data=data)

        fig1.update_layout(
            width=800,
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            title=f'zdt2: {self.input_dimensions}-dim input ',
        )

        fig1.show()

        return self.function
