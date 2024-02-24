from typing import Optional, Union, List

import plotly.graph_objects as go
from scipy.stats import qmc

from functional_tests.blackbox_functions.dtlz2 import DTLZ2
from functional_tests.blackbox_functions.zdt1 import ZDT1
from functional_tests.blackbox_functions.zdt2 import ZDT2
from paref.interfaces.moo_algorithms.paref_moo import ParefMOO
from paref.interfaces.moo_algorithms.stopping_criteria import StoppingCriteria
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import SequenceParetoReflections


class TestingGPRBasedMOOs:
    def __init__(self,
                 stopping_criteria: StoppingCriteria,
                 test_function: str = 'zdt1',
                 sequence_pareto_reflection: Optional[Union[SequenceParetoReflections, ParetoReflection]] = None,
                 input_dimensions: int = 5,
                 max_iter_minimizer: int = 100,
                 lh_evaluations: int = 20,):
        self.input_dimensions = input_dimensions
        self.max_iter_minimizer = max_iter_minimizer
        self.lh_evaluations = lh_evaluations
        self.sequence = sequence_pareto_reflection
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

        # LH evaluation
        [self.function(x) for x in qmc.scale(
            qmc.LatinHypercube(d=self.function.dimension_design_space).random(n=lh_evaluations),
            self.function.design_space.lower_bounds,
            self.function.design_space.upper_bounds,
        )]

        self.real_PF = self.function.return_true_pareto_front()

    def __call__(self,
                 moo: ParefMOO,
                 mark_points: Optional[List] = None,
                 additional_traces: Optional[List[go.Scatter]] = None,
                 ):

        self.lh_evaluations = len(self.function.evaluations)
        if self.sequence is None:
            moo(blackbox_function=self.function,
                stopping_criteria=self.stopping_criteria)
        else:
            moo.apply_to_sequence(blackbox_function=self.function,
                                  sequence_pareto_reflections=self.sequence,
                                  stopping_criteria=self.stopping_criteria)

        y = self.function.y
        if self.function.dimension_target_space == 2:
            data = [
                go.Scatter(x=self.real_PF.T[0],
                           y=self.real_PF.T[1],
                           line=dict(width=4),
                           name='Real Pareto front'),
                go.Scatter(x=y[self.lh_evaluations:].T[0],
                           y=y[self.lh_evaluations:].T[1],
                           mode='markers',
                           marker=dict(size=10),
                           name='Evaluations'),
                go.Scatter(x=y[:self.lh_evaluations].T[0],
                           y=y[:self.lh_evaluations].T[1],
                           mode='markers',
                           name='Initial Evaluations'),
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
                    go.Scatter3d(x=self.function.y[self.lh_evaluations:].T[0],
                                 y=self.function.y[self.lh_evaluations:].T[1],
                                 z=self.function.y[self.lh_evaluations:].T[2],
                                 name='Determined Pareto points',
                                 mode='markers'),
                    go.Scatter3d(x=self.function.y[:self.lh_evaluations].T[0],
                                 y=self.function.y[:self.lh_evaluations].T[1],
                                 z=self.function.y[:self.lh_evaluations].T[2],
                                 name='Initial Evaluations',
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
                x=0.1,
                y=0.9, )
        )

        fig1.show()
        # fig1.write_image(f'../../../docs/graphics/plots/moo-algorithms/{type(moo).__name__}.svg')

        return self.function
