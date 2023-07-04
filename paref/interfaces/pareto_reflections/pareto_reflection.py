from abc import abstractmethod

import numpy as np


class ParetoReflection:
    """Interface for Pareto reflections

    Implement a Pareto reflection

    .. math::

        p: \mathbb{R}^n \\to \mathbb{R}^m

    with this interface.

    Documentation should contain:

    When to use
    -----------
    This sequence should be used if...

    What it does
    ------------
    The sequence...

    Examples
    --------

    TBA: into contribution
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call Pareto reflection to input
        Parameters
        ----------
        x : np.ndarray
            input to witch Pareto reflection is applied

        Returns
        -------
        np.ndarray
            output of Pareto reflection applied to input

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension_codomain(self) -> int:
        """Dimension of codomain of Pareto reflection

        Returns
        -------
        int
            dimension of codomain of Pareto reflection

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension_domain(self) -> int:
        """Dimension of domain of Pareto reflection

        Returns
        -------
        int
            dimension of domain of Pareto reflection

        """
        raise NotImplementedError
