from abc import abstractmethod

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflecting_function import ParetoReflectingFunction


class SequenceParetoReflectingFunctions:
    """Interface fot sequences of Pareto reflections

    Documentation should contain:

    When to use
    -----------
    This Pareto reflection should be used if...

    What it does
    ------------
    The Pareto points of this map are...

    Mathematical formula
    --------------------

    Examples
    --------
    """

    @abstractmethod
    def next(self, blackbox_function: BlackboxFunction) -> ParetoReflectingFunction:
        pass
