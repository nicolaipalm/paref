from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.moo_algorithms.stopping_criteria import StoppingCriteria


class MaxIterationsReached(StoppingCriteria):
    """Stopping criteria based on a maximal number of iterations

    If a defined maximum of iterations is reached, this stopping criteria is met.

    Examples
    --------

    Initialze stopping criteria

    >>> stopping_criteria = MaxIterationsReached(max_iterations=1)

    Evaluate stopping criteria - since this is the first iteration the stopping criteria is not met

    >>> stopping_criteria()
    False

    Since the max iterations (=1) are reached the stopping criteria is met

    >>> stopping_criteria()
    True

    """

    def __init__(self, max_iterations: int = 50):
        """

        Parameters
        ----------
        max_iterations : int
            maximum number of iterations
        """
        self._iteration_step = 0
        self._max_iterations = max_iterations

    def __call__(self, blackbox_function: BlackboxFunction) -> bool:
        """

        Returns
        -------
        bool
            true if the maximal iterations are reached and false otherwise


        """
        if self._iteration_step < self._max_iterations:
            self._iteration_step += 1
            return False

        else:
            return True
