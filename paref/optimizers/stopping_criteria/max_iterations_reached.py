from paref.interfaces.sequences_pareto_reflections.stopping_criteria import StoppingCriteria


class MaxIterationsReached(StoppingCriteria):
    """Stopping criteria based on a maximal number of iterations

    """
    def __init__(self, max_iterations: int = 50):
        """

        Parameters
        ----------
        max_iterations :
        int
            maximal number of iterations
        """
        self._iteration_step = 0
        self._max_iterations = max_iterations

    def __call__(self):
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
