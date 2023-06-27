import numpy as np

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.moo_algorithms.stopping_criteria import StoppingCriteria


class ConvergenceReached(StoppingCriteria):
    """Stopping criteria based on the (2-)distance of the previous evaluations

    compare the previous two evaluations, calculate their (2-)distance and stop when the distance is smaller or equal
    than a certain epsilon.
    For example, if the previous evaluations of the blackbox function are (1,1) and (1,1.1) and the threshold is 0.1,
    then, the stopping criteria returns true since the (2-)distance is 0.1<=epsilon.


    Examples
    --------
    # TBA: add
    """

    def __init__(self,
                 epsilon: float = 5e-2,
                 ):
        """

        Parameters
        ----------

        epsilon : float, default 1e-3
            minimal distance until stopping criteria is reached
        """
        self._epsilon = epsilon

    def __call__(self, blackbox_function: BlackboxFunction) -> bool:
        """

        Returns
        -------
        bool
            true if the 2-norm of the previous two evaluations of the blackbox function is smaller or equal epsilon and
            false otherwise

        """
        if len(blackbox_function.evaluations) >= 2:
            norm = np.linalg.norm(blackbox_function.evaluations[-1][1] -
                                  blackbox_function.evaluations[-2][1])

            if norm > self._epsilon:
                return False

            else:
                print(f'Convergence reached. The 2-distance of the last two points found is {norm}.')
                return True

        else:
            return False
