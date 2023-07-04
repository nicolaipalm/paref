from typing import List

from paref.black_box_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.moo_algorithms.paref_moo import ParefMOO, CompositionWithParetoReflection
from paref.moo_algorithms.minimizer.gpr_minimizer import DifferentialEvolution


class DifferentialEvolutionMinimizer(ParefMOO):
    def apply_moo_operation(self, blackbox_function: BlackboxFunction) -> None:
        minimizer = DifferentialEvolution()
        underlying_blackbox_function = blackbox_function
        while isinstance(underlying_blackbox_function, CompositionWithParetoReflection):
            underlying_blackbox_function = underlying_blackbox_function._blackbox_function

        length_evaluations = len(underlying_blackbox_function.evaluations)
        if isinstance(blackbox_function.design_space, Bounds):
            print('Starting minimization...')
            res = minimizer(
                function=lambda x: blackbox_function(x),
                upper_bounds=blackbox_function.design_space.upper_bounds,
                lower_bounds=blackbox_function.design_space.lower_bounds,
            )
        else:
            raise ValueError('Design space property of blackbox function must be an instance of Bounds!')

        print('finished!')
        underlying_blackbox_function.evaluations = underlying_blackbox_function.evaluations[:length_evaluations]
        underlying_blackbox_function(res)
        print('Value of blackbox: ', underlying_blackbox_function.y[-1])

    @property
    def supported_codomain_dimensions(self) -> List[int]:
        return [1]
