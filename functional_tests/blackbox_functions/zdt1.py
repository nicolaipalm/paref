import numpy as np

from paref.blackbox_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from pymoo.problems import get_problem


class ZDT1(BlackboxFunction):
    def __init__(self, input_dimensions: int = 5,
                 scaling: np.ndarray = np.array([1, 1]),
                 shift: np.ndarray = np.array([0, 0]),
                 ):
        self._input_dimensions = input_dimensions
        self.problem = get_problem('zdt1', n_var=input_dimensions)
        self.bounds = Bounds(upper_bounds=np.ones(input_dimensions), lower_bounds=np.zeros(input_dimensions))
        self.shift = shift
        self.scaling = scaling

    def __call__(self, x):
        return self.problem.evaluate(x) * self.scaling + self.shift

    @property
    def dimension_design_space(self) -> int:
        return self._input_dimensions

    @property
    def dimension_target_space(self) -> int:
        return 2

    @property
    def design_space(self) -> Bounds:
        return self.bounds

    def return_true_pareto_front(self, ):
        return self.problem.pareto_front() * self.scaling + self.shift
