import numpy as np
from scipy.optimize import differential_evolution

from interfaces.minimizer import Minimizer


class DifferentialEvolution(Minimizer):
    """
    ...for harder to minimize functions. Needs more evaluations though.
    """

    def __init__(self, starting_point: np.ndarray,
                 max_iter: int = 500,
                 bounds=None,
                 display=True):
        self.starting_point = starting_point
        self.max_iter = max_iter
        self.bounds = bounds
        self.display = display
        self._number_evaluations_last_call = None

    def __call__(self,
                 function,
                 ):
        if self.bounds == None:
            print('You need to define the bounds...')

        res = differential_evolution(func=function,
                                     x0=self.starting_point,
                                     disp=self.display,
                                     tol=1e-5,
                                     bounds=self.bounds,
                                     maxiter=self.max_iter
                                     )

        self.result = res
        self._number_evaluations_last_call = res.nfev
        return res.x

    @property
    def number_evaluations_last_call(self):
        return self._number_evaluations_last_call
