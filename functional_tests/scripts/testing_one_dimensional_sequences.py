from typing import Union, Optional, List

import plotly.graph_objects as go

from functional_tests.blackbox_functions.dtlz2 import DTLZ2
from functional_tests.blackbox_functions.zdt1 import ZDT1
from functional_tests.blackbox_functions.zdt2 import ZDT2
from paref.interfaces.moo_algorithms.stopping_criteria import StoppingCriteria
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections
from paref.moo_algorithms.minimizer.differential_evolution_minimizer import DifferentialEvolutionMinimizer


class TestingOneDimensionalSequences:
    def __init__(self,
                 stopping_criteria: StoppingCriteria,
                 test_function: str = 'zdt2',
                 input_dimensions: int = 5,):
        self.input_dimensions = input_dimensions
        self.stopping_criteria = stopping_criteria

        test_function_set = ['zdt1', 'zdt2']

        if test_function == 'zdt2':
            self.function = ZDT2(input_dimensions=input_dimensions)

        elif test_function == 'zdt1':
            self.function = ZDT1(input_dimensions=input_dimensions)

        elif test_function == 'dtlz2':
            self.function = DTLZ2(input_dimensions=input_dimensions)

        else:
            raise ValueError(f'Test function must be one of {test_function_set}!')

        self.real_PF = self.function.return_true_pareto_front()

    def __call__(self,
                 sequence: Union[SequenceParetoReflections, ParetoReflection],
                 mark_points: Optional[List] = None,
                 additional_traces: Optional[List[go.Scatter]] = None, ):
        moo = DifferentialEvolutionMinimizer()
        moo.apply_to_sequence(blackbox_function=self.function,
                              sequence_pareto_reflections=sequence,
                              stopping_criteria=self.stopping_criteria)

        y = self.function.y

        if self.function.dimension_target_space == 2:
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

        elif self.function.dimension_target_space == 3:
            pf = self.real_PF
            data = [go.Mesh3d(x=pf.T[0],
                              y=pf.T[1],
                              z=pf.T[2],
                              opacity=0.5,
                              name='Real Pareto front',
                              color='rgba(244,22,100,0.6)'
                              ),
                    go.Scatter3d(x=self.function.y.T[0],
                                 y=self.function.y.T[1],
                                 z=self.function.y.T[2],
                                 name='Determined Pareto points',
                                 mode='markers')
                    ]
            if mark_points is not None:
                data.append(go.Scatter3d(x=mark_points[1].T[0],
                                         y=mark_points[1].T[1],
                                         z=mark_points[1].T[2],
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
