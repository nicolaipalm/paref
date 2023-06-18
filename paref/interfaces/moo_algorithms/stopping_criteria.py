from abc import abstractmethod

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction


class StoppingCriteria:
    """Interface for stopping criteria

    Stopping criteria is exclusively used to indicate/check the end of a sequence of Pareto reflections

    """
    @abstractmethod
    def __call__(self, blackbox_function: BlackboxFunction) -> bool:
        """
        Parameters
        ----------
        blackbox_function : BlackboxFunction
            the blackbox function to which the MOO algorithm is applied

        Returns
        -------
        bool
            true if stopping criteria is met and false otherwise


        """
        pass
