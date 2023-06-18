from abc import abstractmethod

import numpy as np


class ParetoReflection:
    """Interface for Pareto reflections

    Documentation should contain:

    When to use
    -----------
    This sequence should be used if...

    What it does
    ------------
    The sequence...

    Examples
    --------

    TODO: into contribution
    # TODO: include dimension target space - for composition
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension_codomain(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension_domain(self) -> int:
        raise NotImplementedError
