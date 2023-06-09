from abc import abstractmethod

from paref.interfaces.sequences_pareto_reflections.stopping_criteria import StoppingCriteria


class LogicalOrStoppingCriteria(StoppingCriteria):
    """Joining two stopping criteria by a logical or

    Given two stopping criteria, define a new stopping criteria by returning true if at least one of the criteria is met
    (i.e. true).

    """
    def __init__(self,
                 stopping_criteria_1: StoppingCriteria,
                 stopping_criteria_2: StoppingCriteria):
        """

        Parameters
        ----------
        stopping_criteria_1 : StoppingCriteria
            first stopping criteria

        stopping_criteria_2 : StoppingCriteria
            second stopping criteria
        """
        self.stopping_criteria_1 = stopping_criteria_1
        self.stopping_criteria_2 = stopping_criteria_2

    @abstractmethod
    def __call__(self) -> bool:
        """

        Returns
        -------
        bool
            returns true if at least one stopping criteria returns true and false otherwise

        """
        return self.stopping_criteria_1() or self.stopping_criteria_2()
