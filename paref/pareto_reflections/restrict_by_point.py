import numpy as np

from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection


class RestrictByPoint(ParetoReflection):
    """Restrict the Pareto points in the target space by demanding each component to be lower than some value

    When to use
    -----------
    This Pareto reflection should be used if Pareto points are desired which dominate a certain point, i.e.
    when every component of a found Pareto points :math:`x` must satisfy :math:`x_i\\le r_i` for some
    :math:`r_i \\in \\mathbb{R}`.

    What it does
    ------------
    The Pareto points of this map are all the Pareto points which dominate that point.

    Mathematical formula
    --------------------
    Denote by d the point which should be dominated and n some nadir (i.e. n dominated by all points). Then,

    .. math::
        p(x) = x

    if x dominates or is equal to d and

    .. math::
        p(x) = n

    else.

    Examples
    --------
    Define the nadir and the point which should be dominated

    >>> import numpy as np
    >>> nadir, restricting_point = np.array([3,7]),np.zeros(2)

    Initialize the Restricting

    >>> pareto_reflection = RestrictByPoint(nadir=nadir,restricting_point=restricting_point)

    Calling it to (1,1), i.e. calculating

    .. math::
        p(1,1)=(3,7)

    yields:

    >>> pareto_reflection(np.ones(2))
    array([3, 7])

    """

    def __init__(self,
                 nadir: np.ndarray,
                 restricting_point: np.ndarray, ):
        """Specify the nadir and the to be dominated point

        Parameters
        ----------
        nadir :
        np.ndarray
            nadir (dominated by all points) stored in 1 dimensional array of length n
        restricting_point :
        np.ndarray
            point to be dominated stored in 1 dimensional array of length n
        """
        self.nadir = nadir
        self.restricting_point = restricting_point

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Calculate the RestrictByPoint function

        Parameters
        ----------
        x :
        np.ndarray
            input vector stored in 1 dimensional array of length n

        Returns
        -------
        float
            value of the restricting function

        """
        if len(x.shape) != 1:
            raise ValueError(f'Input x must be of dimension 1! Shape of x is {x.shape}.')

        if x.shape != self.restricting_point.shape:
            raise ValueError(
                f'Shapes don\'t match! Shape of x is {x.shape}, shape of restricting point is '
                f'{self.restricting_point.shape}!')

        if np.any(self.restricting_point < x):
            return self.nadir
        else:
            return x

    @property
    def dimension_codomain(self) -> int:
        return len(self.nadir)

    @property
    def dimension_domain(self) -> int:
        return self.dimension_codomain
