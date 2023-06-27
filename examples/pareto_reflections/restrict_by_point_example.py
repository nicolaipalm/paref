import numpy as np
import plotly.graph_objects as go

from examples.scripts.testing_one_dimensional_sequences import TestingOneDimensionalSequences
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflections.find_edge_points import FindEdgePoints
from paref.pareto_reflections.operations.compose_reflections import ComposeReflections
from paref.pareto_reflections.restrict_by_point import RestrictByPoint

# Meta parameters
epsilon = 1e-1
reference_point = 3 * np.ones(2)
nadir = 10 * np.ones(2)
utopia_point = np.zeros(2)
restricting_point = np.array([0.5, 10])

# stopping criteria of MOO algorithm given by maximum iterations
stopping_criteria = MaxIterationsReached(max_iterations=2)

bench = TestingOneDimensionalSequences(input_dimensions=5,
                                       stopping_criteria=stopping_criteria
                                       )

# Apply MOO
pareto_reflection = RestrictByPoint(nadir=nadir, restricting_point=restricting_point)
sequence = ComposeReflections(pareto_reflection,
                              FindEdgePoints(dimension_domain=2,
                                             dimension=0))

area = np.array([[restricting_point[0], 0], restricting_point])
bench(sequence,
      additional_traces=[
          go.Scatter(x=area.T[0], y=area.T[1], fill='tozerox', mode='none', fillcolor='rgba(255, 0, 0, 0.4)',
                     name='Restricted area'),
      ])
