import numpy as np
from pymoo.problems import get_problem

from paref.blackbox_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction


class DTLZ2(BlackboxFunction):
    def __init__(self,
                 input_dimensions: int = 5,
                 scaling: np.ndarray = np.array([1, 1, 1]),
                 shift: np.ndarray = np.array([0, 0, 0]),
                 ):
        self._input_dimensions = input_dimensions
        self.problem = get_problem('dtlz2', n_var=input_dimensions, n_obj=3)
        self.bounds = Bounds(upper_bounds=np.ones(input_dimensions), lower_bounds=np.zeros(input_dimensions))
        self.scaling = scaling
        self.shift = shift

    def __call__(self, x):
        return self.problem.evaluate(x) * self.scaling + self.shift

    @property
    def dimension_design_space(self) -> int:
        return self._input_dimensions

    @property
    def dimension_target_space(self) -> int:
        return 3

    @property
    def design_space(self) -> Bounds:
        return self.bounds

    def return_true_pareto_front(self, ):
        return self.problem.pareto_front() * self.scaling + self.shift
