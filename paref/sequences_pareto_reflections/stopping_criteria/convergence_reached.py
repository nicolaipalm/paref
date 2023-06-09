import numpy as np

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.sequences_pareto_reflections.stopping_criteria import StoppingCriteria


class ConvergenceReached(StoppingCriteria):
    """Stopping criteria based on the (2-)distance of the previous evaluations

    compare the previous two evaluations, calculate their (2-)distance and stop when the distance is smaller or equal
    than a certain epsilon.
    For example, if the previous evaluations of the blackbox function are (1,1) and (1,1.1) and the threshold is 0.1,
    then, the stopping criteria would return true since the (2-)distance is 0.1<=epsilon.
    """
    def __init__(self,
                 blackbox_function: BlackboxFunction,
                 epsilon: float = 1e-3,
                 ):
        """

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            underlying blackbox function

        epsilon : float, default 1e-3
            minimal distance until stopping criteria is reached
        """
        self._epsilon = epsilon
        self._blackbox_function = blackbox_function

    def __call__(self):
        """

        Returns
        -------
        bool
            true if the 2-norm of the previous two evaluations of the blackbox function is smaller or equal epsilon and
            false otherwise

        """
        norm = np.linalg.norm(self._blackbox_function.evaluations[-1][1] -
                              self._blackbox_function.evaluations[-2][1])

        if norm > self._epsilon:
            return False

        else:
            print(f'Convergence reached. The 2-distance of the last two points found is {norm}.')
            return True
