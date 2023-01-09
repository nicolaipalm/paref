import numpy as np
from pymoo.indicators.hv import Hypervolume
from scipy.stats import qmc
import plotly.graph_objects as go
from paref.function_library.interfaces.function import Function
from paref.function_library.zdt1 import ZDT1
from paref.moos.gpr_minimizer import GPRMinimizer
from paref.moos.helper_functions.return_pareto_front import return_pareto_front
from paref.pareto_reflecting_library.functions.epsilon_avoiding import EpsilonAvoiding
from paref.pareto_reflecting_library.functions.interfaces.pareto_reflecting_function import ParetoReflectingFunction
from paref.pareto_reflecting_library.functions.operations.composing import Composing
from paref.pareto_reflecting_library.functions.weighted_norm_to_utopia import WeightedNormToUtopia
from paref.pareto_reflecting_library.sequences.interfaces.sequence_pareto_reflecting_functions import \
    SequenceParetoReflectingFunctions
from paref.pareto_reflecting_library.sequences.repeating_sequence import RepeatingSequence
from paref.stopping_criteria.convergence_reached import ConvergenceReached
from paref.stopping_criteria.interfaces.logical_or_stopping_criteria import LogicalOrStoppingCriteria
from paref.stopping_criteria.max_iterations_reached import MaxIterationsReached

#########
# Setup #
#########

# define test function
input_dimensions = 10
output_dimensions = 2
lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)
function = ZDT1(input_dimensions=input_dimensions)

# evaluate at initial LH
lh_evaluations = 20
[function(x) for x in qmc.scale(
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
print("Search for 2 Pareto point")
# define variables
utopia_point = np.zeros(output_dimensions)
epsilon_convergence = 1e-2

# define constant sequence of l5-norm to zero (=utopia point)
pareto_reflecting_functions = [WeightedNormToUtopia(utopia_point=utopia_point,
                                                    potency=5 * np.ones(2),
                                                    scalar=np.ones(2))]

sequence = RepeatingSequence(pareto_reflecting_functions=pareto_reflecting_functions,
                             stopping_criteria=MaxIterationsReached(max_iterations=1))

# apply minimizer to sequence
moo(blackbox_function=function,
    pareto_reflecting_sequence=sequence,
    stopping_criteria=MaxIterationsReached(max_iterations=1))

####################
# plot the results #
####################
reference_point = 3 * np.ones(2)
PF = return_pareto_front([point[1] for point in function.evaluations])
real_PF = function.return_pareto_front()
hypervolume_max = function.calculate_hypervolume_of_pareto_front(reference_point=reference_point)
metric = Hypervolume(ref_point=reference_point, normalize=False)
hypervolume_weight = metric.do(PF)

maximal_pareto_points = function.y[-1:]

data = [
    go.Scatter(x=real_PF.T[0], y=real_PF.T[1], name="Real Pareto front"),
    go.Scatter(x=maximal_pareto_points.T[0], y=maximal_pareto_points.T[1], mode="markers",
               marker=dict(
                   color="red", size=8),
               name="Maximal Pareto point"),
    go.Scatter(x=function.y[:lh_evaluations].T[0], y=function.y[:lh_evaluations].T[1], mode="markers",
               name="Initial Evaluations"),
]

fig = go.Figure(data=data)

fig.update_layout(
    width=800,
    height=600,
    plot_bgcolor="rgba(0,0,0,0)",
    title=f"zdt1: {input_dimensions}-dim with rel. HV: {hypervolume_weight / hypervolume_max * 100}%",
)

fig.show()

########################
# find 1 Pareto points #
########################
print("Search for 1 Pareto points")

# define stopping criteria for sequence
stopping_criteria = LogicalOrStoppingCriteria(MaxIterationsReached(max_iterations=20),
                                              ConvergenceReached(
                                                  epsilon=epsilon_convergence))

# define constant sequence of linear function searching for the Pareto point corresponding to the second component
pareto_reflecting_functions = [WeightedNormToUtopia(utopia_point=utopia_point,
                                                    potency=np.ones(2),
                                                    scalar=np.array([0.1, 1])),
                               ]

sequence = RepeatingSequence(pareto_reflecting_functions=pareto_reflecting_functions,
                             stopping_criteria=MaxIterationsReached(max_iterations=5))

# apply minimizer to sequence
moo(blackbox_function=function,
    pareto_reflecting_sequence=sequence,
    stopping_criteria=MaxIterationsReached(max_iterations=1))

# define constant sequence of linear function searching for the Pareto point corresponding to the second component
pareto_reflecting_functions = [WeightedNormToUtopia(utopia_point=utopia_point,
                                                    potency=np.ones(2),
                                                    scalar=np.array([1, 0.1])),
                               ]

sequence = RepeatingSequence(pareto_reflecting_functions=pareto_reflecting_functions,
                             stopping_criteria=MaxIterationsReached(max_iterations=5))

# apply minimizer to sequence
moo(blackbox_function=function,
    pareto_reflecting_sequence=sequence,
    stopping_criteria=MaxIterationsReached(max_iterations=1))

####################
# plot the results #
####################
PF = return_pareto_front([point[1] for point in function.evaluations])
one_pareto_points = function.y[-2:]
data = [
    go.Scatter(x=real_PF.T[0], y=real_PF.T[1], name="Real Pareto front"),
    go.Scatter(x=maximal_pareto_points.T[0], y=maximal_pareto_points.T[1], mode="markers",
               marker=dict(
                   color="red", size=8),
               name="Maximal Pareto point"),
    go.Scatter(x=function.y[:lh_evaluations].T[0], y=function.y[:lh_evaluations].T[1], mode="markers",
               name="Initial Evaluations"),
    go.Scatter(x=one_pareto_points.T[0], y=one_pareto_points.T[1], mode="markers", marker=dict(
        color="purple", size=8),
               name="One Pareto points"),
]

fig = go.Figure(data=data)

fig.update_layout(
    width=800,
    height=600,
    plot_bgcolor="rgba(0,0,0,0)",
    title=f"zdt1: {input_dimensions}-dim with rel. HV: {hypervolume_weight / hypervolume_max * 100}%",
)

fig.show()

######################################################################################################################
# find another 2 Pareto point between the 2 Pareto point and the 1 Pareto point corresponding to the second component#
######################################################################################################################
print("Search for evenly separated Pareto points")


# define constant sequence of linear function searching for the Pareto point corresponding to the second component

class EpsilonAvoidingSequenceOfSpecificPoints(SequenceParetoReflectingFunctions):
    def __init__(self,
                 nadir: np.ndarray,
                 pareto_reflecting_function: ParetoReflectingFunction,
                 avoided_points: np.ndarray,
                 epsilon: float = 0):
        self._nadir = nadir
        self._epsilon = epsilon
        self._pareto_reflecting_function = pareto_reflecting_function
        self._avoided_points = avoided_points

    def next(self, blackbox_function: Function) -> ParetoReflectingFunction:
        return Composing(
            EpsilonAvoiding(nadir=self._nadir,
                            epsilon=self._epsilon,
                            epsilon_avoiding_points=self._avoided_points),
            self._pareto_reflecting_function)


avoided_points = [function.y[lh_evaluations + 1]]
number_points = 30 - len(function.y)

distance = np.linalg.norm(avoided_points[0] - function.y[lh_evaluations]) / (number_points + 1) * 0.8

for _ in range(number_points):
    pareto_reflecting_function = WeightedNormToUtopia(utopia_point=utopia_point,
                                                      potency=np.ones(2),
                                                      scalar=np.array([0.1, 1]))

    sequence = EpsilonAvoidingSequenceOfSpecificPoints(nadir=10 * np.ones(2),
                                                       pareto_reflecting_function=pareto_reflecting_function,
                                                       epsilon=distance,
                                                       avoided_points=np.array(avoided_points))

    # while a pareto point is found:

    # apply minimizer to sequence
    moo(blackbox_function=function,
        pareto_reflecting_sequence=sequence,
        stopping_criteria=MaxIterationsReached(max_iterations=1))
    avoided_points.append(function.y[-1])

####################
# plot the results #
####################
PF = return_pareto_front([point[1] for point in function.evaluations])
evenly_pareto_points = function.y[-7:]

data = [
    go.Scatter(x=real_PF.T[0], y=real_PF.T[1], name="Real Pareto front"),
    go.Scatter(x=maximal_pareto_points.T[0], y=maximal_pareto_points.T[1], mode="markers",
               marker=dict(
                   color="red", size=8),
               name="Maximal Pareto point"),
    go.Scatter(x=function.y[:lh_evaluations].T[0], y=function.y[:lh_evaluations].T[1], mode="markers",
               name="Initial Evaluations"),
    go.Scatter(x=one_pareto_points.T[0], y=one_pareto_points.T[1], mode="markers", marker=dict(
        color="purple", size=8),
               name="One Pareto points"),
    go.Scatter(x=evenly_pareto_points.T[0], y=evenly_pareto_points.T[1], mode="markers", marker=dict(
        color="orange", size=8),
               name="Evenly scanned Pareto points"),
]

fig = go.Figure(data=data)

fig.update_layout(
    width=800,
    height=600,
    plot_bgcolor="rgba(0,0,0,0)",
    title=f"zdt1: {input_dimensions}-dim with rel. HV: {hypervolume_weight / hypervolume_max * 100}%",
)

fig.show()

print("Evaluations used:", len(function.x))
