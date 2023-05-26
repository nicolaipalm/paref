import numpy as np

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from pymoo.factory import get_problem
from pymoo.indicators.hv import Hypervolume


class ZDT2(BlackboxFunction):
    def __init__(self, input_dimensions: int = 5):
        super().__init__()
        self.problem = get_problem('zdt2', n_var=input_dimensions)

    def __call__(self, x):
        self._evaluations.append([x, self.problem.evaluate(x)])
        return self.problem.evaluate(x)

    def return_pareto_front(self, ):
        return self.problem.pareto_front()

    def calculate_hypervolume_of_pareto_front(self, reference_point: np.ndarray):
        metric = Hypervolume(ref_point=reference_point, normalize=False)
        return metric.do(self.problem.pareto_front())
