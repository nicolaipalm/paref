import numpy as np
from scipy.optimize import minimize

from interfaces.minimizer import Minimizer


class SLSQP(Minimizer):
    def __init__(self, starting_point: np.ndarray,
                 max_iter: int = 5000,
                 bounds=None,
                 display=False):
        self.starting_point = starting_point
        self.max_iter = max_iter
        self.bounds = bounds
        self.display = display
        self._number_evaluations_last_call = None

    def __call__(self,
                 function,
                 ):
        res = minimize(function,
                       self.starting_point,
                       method='SLSQP',
                       bounds=self.bounds,
                       options={
                           'disp': self.display,
                           'maxiter': self.max_iter
                       })

        self.result = res
        self._number_evaluations_last_call = res.nfev
        return res.x

    @property
    def number_evaluations_last_call(self):
        return self._number_evaluations_last_call
