from abc import abstractmethod
from typing import Optional

from paref.interfaces.optimizers.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflecting_function import ParetoReflectingFunction


class SequenceParetoReflectingFunctions:
    """Interface fot sequences of Pareto reflections

    Documentation of an implementation of this interface should contain:

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

    #TODO: this should be in a contributing.md

    If the end of a sequence is reached return None
    """

    @abstractmethod
    def next(self,) -> Optional[ParetoReflectingFunction]:
        pass
