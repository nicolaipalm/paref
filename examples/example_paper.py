import numpy as np
import plotly.graph_objects as go
from pymoo.indicators.hv import Hypervolume
from scipy.stats import qmc

from examples.function_library.zdt1 import ZDT1
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import \
    SequenceParetoReflections
from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer
from paref.helper_functions.return_pareto_front import return_pareto_front
from paref.pareto_reflections.avoid_points import AvoidPoints
from paref.pareto_reflections.operations.compose_reflections import ComposeReflections
from paref.pareto_reflections.minimize_weighted_norm_to_utopia import MinimizeWeightedNormToUtopia
from paref.pareto_reflection_sequences.generic.repeating_sequence import RepeatingSequence
from paref.pareto_reflection_sequences.restricting_sequence import RestrictingSequence
from paref.moo_algorithms.stopping_criteria.convergence_reached import ConvergenceReached
from paref.moo_algorithms.stopping_criteria.logical_or_stopping_criteria import LogicalOrStoppingCriteria
from paref.moo_algorithms.stopping_criteria import MaxIterationsReached

#########
# Setup #
#########

# TODO: doesnt work anymore

# define test blackbox_function
input_dimensions = 2
output_dimensions = 2
lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
blackbox_function = ZDT1(input_dimensions=input_dimensions)

# evaluate at initial LH
lh_evaluations = 20
[blackbox_function(x) for x in qmc.scale(
    qmc.LatinHypercube(d=len(lower_bounds_x)).random(n=lh_evaluations),
    lower_bounds_x,
    upper_bounds_x,
)]

# define minimizer
max_evaluations = 20
max_iter_minimizer = 100
moo = GPRMinimizer(upper_bounds=upper_bounds_x,
                   lower_bounds=lower_bounds_x)

#########################
# find a 2 Pareto point #
#########################
print('Search for 2 Pareto point')
# define variables
utopia_point = np.zeros(output_dimensions)
epsilon_convergence = 1e-2

# define constant sequence of l5-norm to zero (=utopia point)
pareto_reflecting_functions = [MinimizeWeightedNormToUtopia(utopia_point=utopia_point,
                                                            potency=5 * np.ones(2),
                                                            scalar=np.ones(2))]

sequence = RepeatingSequence(pareto_reflections=pareto_reflecting_functions)

# apply minimizer to sequence
moo(blackbox_function=blackbox_function,
    pareto_reflecting_sequence=sequence,
    )

####################
# plot the results #
####################
reference_point = 3 * np.ones(2)
PF = return_pareto_front([point[1] for point in blackbox_function.evaluations])
real_PF = blackbox_function.pareto_front()
hypervolume_max = blackbox_function.calculate_hypervolume_of_pareto_front(reference_point=reference_point)
metric = Hypervolume(ref_point=reference_point, normalize=False)
hypervolume_weight = metric.do(PF)

maximal_pareto_points = blackbox_function.y[-1:]

data = [
    go.Scatter(x=real_PF.T[0], y=real_PF.T[1], name='Real Pareto front'),
    go.Scatter(x=maximal_pareto_points.T[0], y=maximal_pareto_points.T[1], mode='markers',
               marker=dict(
                   color='red', size=8),
               name='Maximal Pareto point'),
    go.Scatter(x=blackbox_function.y[:lh_evaluations].T[0], y=blackbox_function.y[:lh_evaluations].T[1], mode='markers',
               name='Initial Evaluations'),
]

fig = go.Figure(data=data)

fig.update_layout(
    width=800,
    height=600,
    plot_bgcolor='rgba(0,0,0,0)',
    title=f'zdt1: {input_dimensions}-dim with rel. HV: {hypervolume_weight / hypervolume_max * 100}%',
)

fig.show()

########################
# find 1 Pareto points #
########################
print('Search for 1 Pareto points')

# define stopping criteria for sequence
stopping_criteria = LogicalOrStoppingCriteria(MaxIterationsReached(max_iterations=20),
                                              ConvergenceReached(
                                                  epsilon=epsilon_convergence, blackbox_function=blackbox_function))

# define constant sequence of linear blackbox_function searching for the Pareto
# point corresponding to the second component
pareto_reflecting_functions = [MinimizeWeightedNormToUtopia(utopia_point=utopia_point,
                                                            potency=np.ones(2),
                                                            scalar=np.array([0.1, 1])),
                               ]

sequence = RepeatingSequence(pareto_reflections=pareto_reflecting_functions)

# apply minimizer to sequence
moo(blackbox_function=blackbox_function,
    pareto_reflecting_sequence=sequence,
    )

# define constant sequence of linear blackbox_function searching for the Pareto point corresponding to the second
# component
pareto_reflecting_functions = [MinimizeWeightedNormToUtopia(utopia_point=utopia_point,
                                                            potency=np.ones(2),
                                                            scalar=np.array([1, 0.1])),
                               ]

sequence = RepeatingSequence(pareto_reflections=pareto_reflecting_functions)

# apply minimizer to sequence
moo(blackbox_function=blackbox_function,
    pareto_reflecting_sequence=sequence,
    )

####################
# plot the results #
####################
PF = return_pareto_front([point[1] for point in blackbox_function.evaluations])
one_pareto_points = blackbox_function.y[-2:]
data = [
    go.Scatter(x=real_PF.T[0], y=real_PF.T[1], name='Real Pareto front'),
    go.Scatter(x=maximal_pareto_points.T[0], y=maximal_pareto_points.T[1], mode='markers',
               marker=dict(
                   color='red', size=8),
               name='Maximal Pareto point'),
    go.Scatter(x=blackbox_function.y[:lh_evaluations].T[0], y=blackbox_function.y[:lh_evaluations].T[1], mode='markers',
               name='Initial Evaluations'),
    go.Scatter(x=one_pareto_points.T[0], y=one_pareto_points.T[1], mode='markers', marker=dict(
        color='purple', size=8),
               name='One Pareto points'),
]

fig = go.Figure(data=data)

fig.update_layout(
    width=800,
    height=600,
    plot_bgcolor='rgba(0,0,0,0)',
    title=f'zdt1: {input_dimensions}-dim with rel. HV: {hypervolume_weight / hypervolume_max * 100}%',
)

fig.show()

###################################
# find Evenly spaced Pareto points#
###################################

# define constant sequence of linear blackbox_function searching for the Pareto point corresponding to the second
# component
pareto_reflecting_function = MinimizeWeightedNormToUtopia(utopia_point=utopia_point,
                                                          potency=np.ones(2),
                                                          scalar=np.array([0.1, 1]))

restricting_point = np.array(
    [0.2, 10])
print(restricting_point)
sequence = RestrictingSequence(nadir=10 * np.ones(2), stopping_criteria=stopping_criteria,
                               pareto_reflecting_function=pareto_reflecting_function,
                               restricting_point=restricting_point)

# apply minimizer to sequence
moo(blackbox_function=blackbox_function,
    pareto_reflecting_sequence=sequence,
    )

# Search for
restricting_point = np.array(
    [0.7, 10])
print(restricting_point)
sequence = RestrictingSequence(nadir=10 * np.ones(2), stopping_criteria=stopping_criteria,
                               pareto_reflecting_function=pareto_reflecting_function,
                               restricting_point=restricting_point)

# apply minimizer to sequence
moo(blackbox_function=blackbox_function,
    pareto_reflecting_sequence=sequence,
    )

####################
# plot the results #
####################
PF = return_pareto_front([point[1] for point in blackbox_function.evaluations])
spaced_pareto_points = blackbox_function.y[-2:]
data = [
    go.Scatter(x=real_PF.T[0], y=real_PF.T[1], name='Real Pareto front'),
    go.Scatter(x=maximal_pareto_points.T[0], y=maximal_pareto_points.T[1], mode='markers',
               marker=dict(
                   color='red', size=8),
               name='Maximal Pareto point'),
    go.Scatter(x=blackbox_function.y[:lh_evaluations].T[0], y=blackbox_function.y[:lh_evaluations].T[1], mode='markers',
               name='Initial Evaluations'),
    go.Scatter(x=one_pareto_points.T[0], y=one_pareto_points.T[1], mode='markers', marker=dict(
        color='purple', size=8),
               name='One Pareto points'),
    go.Scatter(x=spaced_pareto_points.T[0], y=spaced_pareto_points.T[1], mode='markers', marker=dict(
        color='pink', size=8),
               name='Evenly distributed of whole front'),
]

fig = go.Figure(data=data)

fig.update_layout(
    width=800,
    height=600,
    plot_bgcolor='rgba(0,0,0,0)',
    title=f'zdt1: {input_dimensions}-dim with rel. HV: {hypervolume_weight / hypervolume_max * 100}%',
)

fig.show()

############################################################################################################
# find Pareto point between the 2 Pareto point and the 1 Pareto point corresponding to the second component#
############################################################################################################
print('Search for evenly separated Pareto points')


# define constant sequence of linear blackbox_function searching for the Pareto point corresponding to the second
# component

class EpsilonAvoidingSequenceOfSpecificPoints(SequenceParetoReflections):
    def __init__(self,
                 nadir: np.ndarray,
                 pareto_reflecting_function: ParetoReflection,
                 avoided_points: np.ndarray,
                 epsilon: float = 0):
        self._nadir = nadir
        self._epsilon = epsilon
        self._pareto_reflecting_function = pareto_reflecting_function
        self._avoided_points = avoided_points

    def next(self) -> ParetoReflection:
        return ComposeReflections(
            AvoidPoints(nadir=self._nadir,
                        epsilon=self._epsilon,
                        epsilon_avoiding_points=self._avoided_points),
            self._pareto_reflecting_function)


avoided_points = [one_pareto_points[0]]

number_points = 5

distance = np.linalg.norm(one_pareto_points[0] - blackbox_function.y[-1]) / (number_points + 1) * 0.8

for i in range(number_points):
    pareto_reflecting_function = MinimizeWeightedNormToUtopia(utopia_point=utopia_point,
                                                              potency=np.ones(2),
                                                              scalar=np.array([0.1, 1]))

    sequence = EpsilonAvoidingSequenceOfSpecificPoints(nadir=10 * np.ones(2),
                                                       pareto_reflecting_function=pareto_reflecting_function,
                                                       epsilon=distance,
                                                       avoided_points=np.array(avoided_points))
    # apply minimizer to sequence
    moo(blackbox_function=blackbox_function,
        pareto_reflecting_sequence=sequence,
        )

    avoided_points.append(blackbox_function.y[-1])

####################
# plot the results #
####################
PF = return_pareto_front([point[1] for point in blackbox_function.evaluations])
evenly_pareto_points = blackbox_function.y[-number_points:]

data = [
    go.Scatter(x=real_PF.T[0], y=real_PF.T[1], name='Real Pareto front'),
    go.Scatter(x=maximal_pareto_points.T[0], y=maximal_pareto_points.T[1], mode='markers',
               marker=dict(
                   color='red', size=8),
               name='Maximal Pareto point'),
    go.Scatter(x=blackbox_function.y[:lh_evaluations].T[0], y=blackbox_function.y[:lh_evaluations].T[1], mode='markers',
               name='Initial Evaluations'),
    go.Scatter(x=one_pareto_points.T[0], y=one_pareto_points.T[1], mode='markers', marker=dict(
        color='purple', size=8),
               name='One Pareto points'),
    go.Scatter(x=spaced_pareto_points.T[0], y=spaced_pareto_points.T[1], mode='markers', marker=dict(
        color='pink', size=8),
               name='Evenly distributed of whole front'),
    go.Scatter(x=evenly_pareto_points.T[0], y=evenly_pareto_points.T[1], mode='markers', marker=dict(
        color='orange', size=8),
               name='Evenly scanned Pareto points'),
]

fig = go.Figure(data=data)

fig.update_layout(
    width=800,
    height=600,
    plot_bgcolor='rgba(0,0,0,0)',
    title=f'zdt1: {input_dimensions}-dim with rel. HV: {hypervolume_weight / hypervolume_max * 100}%',
)

fig.show()

print('Evaluations used:', len(blackbox_function.x))

number_points = 30 - len(blackbox_function.y)  # = 7

distance = np.linalg.norm(avoided_points[0] - blackbox_function.y[lh_evaluations]) / (number_points + 1) * 0.8
