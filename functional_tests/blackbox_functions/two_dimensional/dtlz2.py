import numpy as np
from pymoo.factory import get_problem
from pymoo.indicators.hv import Hypervolume

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction


class DTLZ2(BlackboxFunction):
    def __init__(self, input_dimensions: int = 5, output_dimensions: int = 2):
        self.problem = get_problem('dtlz2', n_var=input_dimensions, n_obj=output_dimensions)

    def __call__(self, x):
        return self.problem.evaluate(x)

    def return_pareto_front(self, ):
        return self.problem.return_true_pareto_front()

    def calculate_hypervolume_of_pareto_front(self, reference_point: np.ndarray):
        metric = Hypervolume(ref_point=reference_point, normalize=False)
        return metric.do(self.problem.return_true_pareto_front())
