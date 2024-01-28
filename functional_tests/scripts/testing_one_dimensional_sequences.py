from typing import Union, Optional, List

import numpy as np
import plotly.graph_objects as go
from pymoo.indicators.hv import Hypervolume

from functional_tests.blackbox_functions.two_dimensional.zdt1 import ZDT1
from functional_tests.blackbox_functions.two_dimensional.zdt2 import ZDT2
from paref.interfaces.moo_algorithms.stopping_criteria import StoppingCriteria
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.moo_algorithms.minimizer.differential_evolution_minimizer import DifferentialEvolutionMinimizer


class TestingOneDimensionalSequences:
    def __init__(self,
                 stopping_criteria: StoppingCriteria,
                 test_function: str = 'zdt2',
                 input_dimensions: int = 5,
                 reference_point: np.ndarray = np.array([2, 2])):
        self.input_dimensions = input_dimensions
        self.stopping_criteria = stopping_criteria

        test_function_set = ['zdt1', 'zdt2']

        if test_function == 'zdt2':
            self.function = ZDT2(input_dimensions=input_dimensions)

        elif test_function == 'zdt1':
            self.function = ZDT1(input_dimensions=input_dimensions)

        else:
            raise ValueError(f'Test function must be one of {test_function_set}!')

        self.real_PF = self.function.return_true_pareto_front()
        self.hypervolume_max = self.function.calculate_hypervolume_of_pareto_front(reference_point=reference_point)
        self.metric = Hypervolume(ref_point=reference_point, normalize=False)

    def __call__(self,
                 sequence: Union[SequenceParetoReflections, ParetoReflection],
                 mark_points: Optional[List] = None,
                 additional_traces: Optional[List[go.Scatter]] = None, ):
        moo = DifferentialEvolutionMinimizer()
        moo.apply_to_sequence(blackbox_function=self.function,
                              sequence_pareto_reflections=sequence,
                              stopping_criteria=self.stopping_criteria)

        y = np.array([evaluation[1] for evaluation in self.function.evaluations])

        data = [
            go.Scatter(x=self.real_PF.T[0],
                       y=self.real_PF.T[1],
                       name='Real Pareto front',
                       line=dict(width=4)
                       ),
            go.Scatter(x=y.T[0], y=y.T[1],
                       mode='markers',
                       marker=dict(size=10),
                       name='Determined Pareto points'
                       ),
        ]
        if mark_points is not None:
            data.append(go.Scatter(x=mark_points[1].T[0], y=mark_points[1].T[1],
                                   mode='markers',
                                   marker=dict(size=10, symbol='x'),
                                   name=mark_points[0],
                                   )
                        )

        if additional_traces is not None:
            for additional_trace in additional_traces:
                data.append(additional_trace)

        fig1 = go.Figure(data=data)

        fig1.update_layout(
            width=500,
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                x=0.6,
                y=0.9, )
        )

        fig1.show()
        # fig1.write_image(f'../../docs/graphics/plots/reflections/{type(sequence).__name__}.svg')

        return self.function
