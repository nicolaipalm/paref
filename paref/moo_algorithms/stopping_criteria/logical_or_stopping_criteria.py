from abc import abstractmethod

from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.moo_algorithms.stopping_criteria import StoppingCriteria


class LogicalOrStoppingCriteria(StoppingCriteria):
    """Join two stopping criteria by a logical or

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
    def __call__(self, blackbox_function: BlackboxFunction) -> bool:
        """

        Returns
        -------
        bool
            returns true if at least one stopping criteria returns true and false otherwise

        """
        return self.stopping_criteria_1(blackbox_function) or self.stopping_criteria_2(blackbox_function)
