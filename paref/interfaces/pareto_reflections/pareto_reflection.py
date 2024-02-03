from abc import abstractmethod

import numpy as np


class ParetoReflection:
    """Interface for Pareto reflections

    Implement a Pareto reflection

    .. math::

        p: \mathbb{R}^n \\to \mathbb{R}^m

    with this interface.
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

    def best_fits(self, points: np.ndarray) -> np.ndarray:
        """Return the Pareto points of Pareto reflection with respect to the variable points

        Parameters
        ----------
        points : np.ndarray
            Points to which the Pareto reflection is restricted

        Returns
        -------
        np.ndarray
            (Pareto) points of Pareto reflection restricted to points array

        """
        array = [self(point) for point in points]
        pareto_points_indices = []
        for i, point in enumerate(array):
            is_pareto = True
            for j, other in enumerate(array):
                if i == j:
                    continue
                if np.all(point >= other) and np.any(point > other):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_points_indices.append(i)
        return np.unique(np.array([points[i] for i in pareto_points_indices]), axis=0)
