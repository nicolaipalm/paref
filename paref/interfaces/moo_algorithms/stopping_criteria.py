from abc import abstractmethod

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction


class StoppingCriteria:
    """Interface for stopping criteria

    Stopping criteria are used to indicate the end of a sequence of Pareto reflections or the end of an
    MOO algorithm.
    Evaluated, they return true if the stopping criteria is met and false otherwise.

    """
    @abstractmethod
    def __call__(self, blackbox_function: BlackboxFunction) -> bool:
        """
        Parameters
        ----------
        blackbox_function : BlackboxFunction
            the blackbox function to which the MOO algorithm/sequence of Pareto reflections is applied

        Returns
        -------
        bool
            true if stopping criteria is met and false otherwise


        """
        pass
