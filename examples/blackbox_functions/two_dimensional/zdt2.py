import numpy as np

from paref.black_box_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from pymoo.factory import get_problem
from pymoo.indicators.hv import Hypervolume


class ZDT2(BlackboxFunction):
    def __init__(self, input_dimensions: int = 5):
        self._input_dimensions = input_dimensions
        super().__init__()
        self.problem = get_problem('zdt2', n_var=input_dimensions)
        self.bounds = Bounds(upper_bounds=np.ones(input_dimensions), lower_bounds=np.zeros(input_dimensions))

    def __call__(self, x):
        self._evaluations.append([x, self.problem.evaluate(x)])
        return self.problem.evaluate(x)

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
        return self.problem.pareto_front()

    def calculate_hypervolume_of_pareto_front(self, reference_point: np.ndarray):
        metric = Hypervolume(ref_point=reference_point, normalize=False)
        return metric.do(self.problem.pareto_front())
