import numpy as np


class Bounds:
    def __init__(self,
                 upper_bounds: np.ndarray,
                 lower_bounds: np.ndarray):
        if len(upper_bounds.shape) != 1:
            raise ValueError('Upper bounds need to be a single-dimensional array')

        if len(lower_bounds.shape) != 1:
            raise ValueError('Lower bounds need to be a single-dimensional array')
        self._upper_bounds = upper_bounds
        self._lower_bounds = lower_bounds

    @property
    def upper_bounds(self):
        return self._upper_bounds

    @property
    def lower_bounds(self):
        return self._lower_bounds
