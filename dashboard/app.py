import gradio as gr
import numpy as np
import plotly.graph_objects as go
from scipy.stats import qmc

from functional_tests.blackbox_functions.zdt1 import ZDT1
from functional_tests.blackbox_functions.zdt2 import ZDT2
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.moo_algorithms.minimizer.differential_evolution_minimizer import DifferentialEvolutionMinimizer
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflection_sequences.generic.repeating_sequence import RepeatingSequence
from paref.pareto_reflection_sequences.two_dimensional.fill_gaps_of_pareto_front_sequence_2d import \
    FillGapsOfParetoFrontSequence2D
from paref.pareto_reflections.find_1_pareto_points import Find1ParetoPoints
from paref.pareto_reflections.find_maximal_pareto_point import FindMaximalParetoPoint
from paref.pareto_reflections.operations.compose_sequences import ComposeSequences
from paref.pareto_reflections.priority_search import PrioritySearch
from paref.pareto_reflections.restrict_by_point import RestrictByPoint

bbf_names = [
    'ZDT2',
    'ZDT1',
]


class BBFs:
    def __init__(self):
        self.bbfs = {
            'ZDT2': ZDT2(input_dimensions=4),
            'ZDT1': ZDT1(input_dimensions=4),
        }
        self.res_point = RestrictByPoint(nadir=np.array([10, 10]), restricting_point=np.array([10, 10]))

    def select_bbf(self, bbf_name) -> BlackboxFunction:
        return self.bbfs[bbf_name]

    def set_bbf(self, bbf_name, restricting_point):
        self.res_point = RestrictByPoint(nadir=np.array([10, 10]), restricting_point=restricting_point)
        bbf = self.bbfs[bbf_name]
        bbf.clear_evaluations()
        area = np.array([[self.res_point.restricting_point[0], 0], self.res_point.restricting_point])
        data = [
            go.Scatter(x=bbf.return_true_pareto_front().T[0], y=bbf.return_true_pareto_front().T[1],
                       marker=dict(size=10),
                       name='True Pareto front'
                       ),
            go.Scatter(x=area.T[0], y=area.T[1], fill='tozerox', mode='none', fillcolor='rgba(72,118,255, 0.2)',
                       name='Allowed area'),
        ]

        fig = go.Figure(data=data)

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                x=0.6,
                y=0.9, )
        )
        fig.update_yaxes(range=[-0.01, 1.5])
        fig.update_xaxes(range=[-0.01, 1.5])

        return fig

    def lh_search(self, bbf_name, number_evaluations):
        bbf = self.bbfs[bbf_name]
        number_current_evaluations = len(bbf.y)
        [self.bbfs[bbf_name](x) for x in qmc.scale(
            qmc.LatinHypercube(d=bbf.dimension_design_space).random(
                n=number_evaluations),
            self.bbfs[bbf_name].design_space.lower_bounds,
            self.bbfs[bbf_name].design_space.upper_bounds,
        )]  # add samples according to latin hypercube scheme
        area = np.array([[self.res_point.restricting_point[0], 0], self.res_point.restricting_point])
        data = [
            go.Scatter(x=bbf.return_true_pareto_front().T[0], y=bbf.return_true_pareto_front().T[1],
                       marker=dict(size=10),
                       name='True Pareto front'
                       ),
            go.Scatter(x=bbf.y.T[0], y=bbf.y.T[1],
                       mode='markers',
                       marker=dict(size=10),
                       name='Determined points'
                       ),
            go.Scatter(x=bbf.y[number_current_evaluations:].T[0], y=bbf.y[number_current_evaluations:].T[1],
                       mode='markers',
                       marker=dict(size=10),
                       name='LHC samples'
                       ),
            go.Scatter(x=area.T[0], y=area.T[1], fill='tozerox', mode='none', fillcolor='rgba(72,118,255, 0.2)',
                       name='Allowed area'),
        ]

        fig = go.Figure(data=data)

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                x=0.6,
                y=0.9, )
        )

        fig.update_yaxes(range=[-0.01, 8])
        fig.update_xaxes(range=[-0.01, 1.5])

        return fig

    def run_moo(self, pareto_reflection, bbf_name, iterations=1):
        pareto_reflection = ComposeSequences(self.res_point, pareto_reflection)

        bbf = self.bbfs[bbf_name]
        moo = DifferentialEvolutionMinimizer()
        moo.apply_to_sequence(blackbox_function=bbf,
                              sequence_pareto_reflections=pareto_reflection,
                              stopping_criteria=MaxIterationsReached(max_iterations=iterations))
        # Plot evals
        area = np.array([[self.res_point.restricting_point[0], 0], self.res_point.restricting_point])
        data = [
            go.Scatter(x=bbf.return_true_pareto_front().T[0], y=bbf.return_true_pareto_front().T[1],
                       marker=dict(size=10),
                       name='True Pareto front'
                       ),
            go.Scatter(x=bbf.y.T[0], y=bbf.y.T[1],
                       mode='markers',
                       marker=dict(size=10),
                       name='Determined points'
                       ),
            go.Scatter(x=bbf.y[-iterations:].T[0], y=bbf.y[-iterations:].T[1],
                       mode='markers',
                       marker=dict(size=10),
                       name='Pareto point from recent MOO'
                       ),
            go.Scatter(x=area.T[0], y=area.T[1], fill='tozerox', mode='none', fillcolor='rgba(72,118,255, 0.2)',
                       name='Allowed area'),
        ]

        fig = go.Figure(data=data)

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                x=0.6,
                y=0.9, )
        )
        fig.update_yaxes(range=[-0.01, 1.5])
        fig.update_xaxes(range=[-0.01, 1.5])

        return fig


css = """
h1 {
    text-align: center;
    display:block;
}
"""
with gr.Blocks(css=css) as demo:
    gr.Markdown('# Paref Showcase')
    gr.Markdown("""Welcome 👋! This is a showcase of the [Paref package]().
    Here you can interactively explore the different Pareto reflections and multi-objective optimization algorithms
    Paref provides. Its purpose is to help you understand what Pareto point each algorithm/Pareto reflection
    looks for. Start by selecting a blackbox function and possible constraints and click the Set Blackbox Function
    button.
    """)
    model = BBFs()
    initial_bbf_name = 'ZDT2'

    with gr.Column():
        with gr.Row():
            bbf_name = gr.Dropdown(bbf_names, label='Blackbox Function', value=initial_bbf_name)
            with gr.Row():
                constraint_y1 = gr.Number(value=10, label='Constraint y1 smaller than', minimum=0.2, maximum=10)
                btn_set_bbf = gr.Button(value='Set Blackbox Function')

    with gr.Row(equal_height=False):
        plot = gr.Plot(model.set_bbf(initial_bbf_name, np.array([10, 10])))
        btn_set_bbf.click(lambda x, y: model.set_bbf(x, np.array([y, 10])), [bbf_name, constraint_y1],
                          plot)

        with gr.Column():
            with gr.Accordion('express.minimal_search', open=True):
                with gr.Row(equal_height=True):
                    gr.Markdown(label='Description', value='Apply Paref Express minimal search algorithm.'
                                                           'This algorithm yields Pareto points minimal '
                                                           'in some component and '
                                                           'a `real trade-off` between all components.')
                btn = gr.Button(value='Run MOO')

                btn.click(
                    lambda x: model.run_moo(RepeatingSequence([
                        Find1ParetoPoints(blackbox_function=model.bbfs[x], dimension=0),
                        Find1ParetoPoints(blackbox_function=model.bbfs[x], dimension=1),
                        FindMaximalParetoPoint(blackbox_function=model.bbfs[x]),
                    ]), x, 3),
                    [bbf_name], plot)

            with gr.Accordion('express.priority_search', open=False):
                with gr.Row(equal_height=True):
                    gr.Markdown(label='Description',
                                value='Find the Pareto point which represents your weights, i.e. priority '
                                      'of certain components. For example, component one is more important'
                                      'to you than component two. '
                                      'You quantify that importance by assigning '
                                      'component one a weight of 80 and component two a weight of 20. '
                                      '\n**WARNING**: This Pareto reflection assumes that the '
                                      'minima of components '
                                      'was already (approximately) found. '
                                      'Apply the find_1_pareto_points reflection'
                                      'in both dimensions first!')
                    with gr.Column():
                        priority_1 = gr.Number(value=80, label='Weight on 1th component', minimum=0, maximum=100)
                        priority_2 = gr.Number(value=20, label='Weight on 2th component', minimum=0, maximum=100)
                btn = gr.Button(value='Run MOO')
                btn.click(
                    lambda x, y, z: model.run_moo(
                        PrioritySearch(blackbox_function=model.bbfs[x], priority=np.array([y, z])), x),
                    [bbf_name, priority_1, priority_2], plot)

            with gr.Accordion('Latin Hypercube random search', open=False):
                with gr.Row(equal_height=True):
                    gr.Markdown(label='Description', value="""
                    Conduct a random search based on the Latin Hypercube Sampling (LHS) method.
                    The LHS is by default the initial sampling method for every MOO algorithm in Paref.
                    """)
                    number_evaluations = gr.Number(value=20, label='Number of evaluations', minimum=1, maximum=100)

                btn = gr.Button(value='Run LHS')
                btn.click(lambda x, y: model.lh_search(x, y), [bbf_name, number_evaluations], plot)

            with gr.Accordion('find_1_pareto_points', open=False):
                with gr.Row(equal_height=True):
                    gr.Markdown(label='Description', value='Determine the Pareto point '
                                                           'which is minimal in the specified dimension.')
                    with gr.Column():
                        dimension = gr.Number(value=0, label='Dimension', minimum=0, maximum=1)
                btn = gr.Button(value='Run MOO')
                btn.click(
                    lambda x, y: model.run_moo(Find1ParetoPoints(blackbox_function=model.bbfs[x], dimension=y), x),
                    [bbf_name, dimension], plot)

            with gr.Accordion('fill_gap', open=False):
                with gr.Row(equal_height=True):
                    gr.Markdown(label='Description', value='Determine the Pareto point '
                                                           'which fills the gap between target space '
                                                           'dimension(=2 in this example) many points.'
                                                           'In this example the largest gap is automatically chosen.'
                                                           '\n**WARNING**: '
                                                           'you need at least two Pareto points prior to this!')
                btn = gr.Button(value='Run MOO')
                btn.click(
                    lambda x: model.run_moo(FillGapsOfParetoFrontSequence2D(), x),
                    [bbf_name], plot)

            with gr.Accordion('find_maximal_pareto_point', open=False):
                with gr.Row(equal_height=True):
                    gr.Markdown(label='Description', value='Determine a maximal Pareto point, i.e. '
                                                           'some Pareto point which represents no minimum in '
                                                           'any dimension,'
                                                           'in other words, which is a true trade-off between '
                                                           'all components.'
                                                           '\n**WARNING**: This Pareto reflection assumes that the '
                                                           'minima of components '
                                                           'was already (approximately) found. '
                                                           'Apply the find_1_pareto_points reflection'
                                                           'in both dimensions first!')
                btn = gr.Button(value='Run MOO')
                btn.click(
                    lambda x: model.run_moo(FindMaximalParetoPoint(blackbox_function=model.bbfs[x]), x),
                    [bbf_name], plot)

demo.launch()
