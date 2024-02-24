from typing import List

from paref.blackbox_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.moo_algorithms.paref_moo import ParefMOO, CompositionWithParetoReflection
from paref.moo_algorithms.minimizer.gpr_minimizer import DifferentialEvolution
from paref.moo_algorithms.multi_dimensional.min_g import calculate_optimal_scaling_x, calculate_optimal_scaling_g
from paref.pareto_reflections.minimize_g import MinGParetoReflection
from paref.pareto_reflections.operations.compose_reflections import ComposeReflections


class DifferentialEvolutionMinimizer(ParefMOO):
    def __init__(self):
        self._minimizer = DifferentialEvolution()
        self._scaling_x = None

    def apply_moo_operation(self, blackbox_function: BlackboxFunction) -> None:
        if not isinstance(blackbox_function.design_space, Bounds):
            raise ValueError('Design space property of blackbox function must be an instance of Bounds!')

        base_blackbox_function = blackbox_function
        length_evaluations = len(base_blackbox_function.evaluations)

        #####
        # extract Pareto reflections
        pareto_reflections = []
        while isinstance(base_blackbox_function, CompositionWithParetoReflection):
            pareto_reflections.append(base_blackbox_function._pareto_reflection)
            base_blackbox_function = base_blackbox_function._blackbox_function

        # compute optimal scaling whenever MinGParetoReflection is used

        if isinstance(pareto_reflections[0], MinGParetoReflection):
            print('Calculating optimal scaling...')
            if len(pareto_reflections) > 1:
                pareto_reflection = pareto_reflections[1]
                for reflection in pareto_reflections[2:]:
                    pareto_reflection = ComposeReflections(reflection, pareto_reflection)
                base_fun = lambda x: pareto_reflection(base_blackbox_function(x))
            else:
                base_fun = lambda x: base_blackbox_function(x)

            if self._scaling_x is None:
                self._scaling_x = calculate_optimal_scaling_x(base_fun, blackbox_function)
            pareto_reflections[0].g.scaling_x = self._scaling_x
            pareto_reflections[0].g.scaling_g = calculate_optimal_scaling_g(base_fun, pareto_reflections[0].g,
                                                                            blackbox_function)
            fun = lambda x: pareto_reflections[0](base_fun(x))

        else:
            fun = blackbox_function

        if isinstance(blackbox_function.design_space, Bounds):
            print('Starting optimization...')
            res = self._minimizer(
                function=lambda x: fun(x),
                upper_bounds=blackbox_function.design_space.upper_bounds,
                lower_bounds=blackbox_function.design_space.lower_bounds,
            )
        else:
            raise ValueError('Design space property of blackbox function must be an instance of Bounds!')

        print('finished!')
        base_blackbox_function.evaluations = base_blackbox_function.evaluations[:length_evaluations]
        base_blackbox_function(res)
        # print('Value of blackbox: ', base_blackbox_function.y[-1])

    @property
    def supported_codomain_dimensions(self) -> List[int]:
        return [1]
