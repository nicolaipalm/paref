from abc import abstractmethod


class StoppingCriteria:
    """Interface for stopping criteria

    Stopping criteria is exclusively used to indicate/check the end of a sequence of Pareto reflections

    """
    @abstractmethod
    def __call__(self, ) -> bool:
        """

        Returns
        -------
        bool
            true if stopping criteria is met and false otherwise


        """
        pass
