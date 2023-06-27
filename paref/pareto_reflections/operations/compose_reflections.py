import numpy as np

from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection


# TBA: as method of reflections

class ComposeReflections(ParetoReflection):
    """Compose two Pareto reflections and obtain a new Pareto reflection

    """

    def __init__(self,
                 pareto_reflecting_function_1: ParetoReflection,
                 pareto_reflecting_function_2: ParetoReflection):
        """

        Parameters
        ----------
        pareto_reflecting_function_1 : ParetoReflection
            Pareto reflection which is applied first

        pareto_reflecting_function_2 : ParetoReflection
            Pareto reflection which is applied second
        """
        self.pareto_reflecting_function_1 = pareto_reflecting_function_1
        self.pareto_reflecting_function_2 = pareto_reflecting_function_2

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        x : np.ndarray
            input of Pareto reflection

        Returns
        -------
        np.ndarray
            value of the composition at x, i.e. pareto_reflecting_function_2(pareto_reflecting_function_1(x))

        """
        return self.pareto_reflecting_function_2(self.pareto_reflecting_function_1(x))

    @property
    def dimension_codomain(self) -> int:
        return self.pareto_reflecting_function_1.dimension_codomain

    @property
    def dimension_domain(self) -> int:
        return self.pareto_reflecting_function_2.dimension_domain
