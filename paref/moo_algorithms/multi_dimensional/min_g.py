from abc import abstractmethod
from time import sleep
from warnings import warn

import numpy as np

from paref.blackbox_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.moo_algorithms.paref_moo import CompositionWithParetoReflection
from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer, DifferentialEvolution
from paref.moo_algorithms.minimizer.surrogates.gpr import GPR
from paref.pareto_reflections.operations.compose_reflections import ComposeReflections


def calculate_optimal_scaling_x(fun, blackbox_function, max_iter_minimizer: int = 500):
    # scale each component to [0,1]
    dim_f = len(fun(blackbox_function.design_space.upper_bounds))
    minimizer = DifferentialEvolution()
    x_min = np.zeros(dim_f)
    x_max = np.zeros(dim_f)
    for i in range(len(x_min)):
        res_i_min = minimizer(
            function=lambda x: fun(x)[i],
            max_iter=max_iter_minimizer,
            upper_bounds=blackbox_function.design_space.upper_bounds,
            lower_bounds=blackbox_function.design_space.lower_bounds,
        )
        res_i_max = minimizer(
            function=lambda x: -fun(x)[i],
            max_iter=max_iter_minimizer,
            upper_bounds=blackbox_function.design_space.upper_bounds,
            lower_bounds=blackbox_function.design_space.lower_bounds,
        )
        x_min[i] = fun(res_i_min)[i]
        x_max[i] = fun(res_i_max)[i]

    return lambda x: (x - x_min) / (x_max - x_min)


def calculate_optimal_scaling_g(fun, g, blackbox_function, max_iter_minimizer: int = 500):
    # Scale g and each component to [0,1]
    minimizer = DifferentialEvolution()
    res_g = minimizer(
        function=lambda x: g(fun(x)),
        max_iter=max_iter_minimizer,
        upper_bounds=blackbox_function.design_space.upper_bounds,
        lower_bounds=blackbox_function.design_space.lower_bounds,
    )

    res_g_max = minimizer(
        function=lambda x: -g(fun(x)),
        max_iter=max_iter_minimizer,
        upper_bounds=blackbox_function.design_space.upper_bounds,
        lower_bounds=blackbox_function.design_space.lower_bounds,
    )

    xg_min = fun(res_g)
    gxg_min = g(xg_min)
    xg_max = fun(res_g_max)
    gxg_max = g(xg_max)
    return lambda x: (x - gxg_min) / (gxg_max - gxg_min)


class MinG(GPRMinimizer):
    @property
    @abstractmethod
    def sequence_of_pareto_reflections(self):
        pass

    def apply_moo_operation(self,
                            blackbox_function: BlackboxFunction,
                            ) -> None:
        """Apply moo operation constructed as above

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            blackbox function to which algorithm is applied


        """
        # TBA: when found points are too close stop!
        # TBA: control mechanism: when algo doesn't work give message about what went wrong
        # TBA: monitoring: stop time, evaluations found, if training process of gpr converged, all with hints

        if len(blackbox_function.y) < 20:
            raise ValueError('Blackbox function must have at least 20 evaluations! Apply the latin hypercube sampling '
                             '(blackbox_function.perform_lhc(n=20)) first!')

        gpr = GPR(training_iter=self._training_iter, learning_rate=self._learning_rate, )

        base_blackbox_function = blackbox_function

        pareto_reflections = []
        while isinstance(base_blackbox_function, CompositionWithParetoReflection):
            pareto_reflections.append(base_blackbox_function._pareto_reflection)
            base_blackbox_function = base_blackbox_function._blackbox_function

        train_x = base_blackbox_function.x
        train_y = base_blackbox_function.y
        print('\n=========================================================='
              '\n==========================================================')
        print(
            'Training...\n'
        )
        sleep(0.1)  # ensure that the print statement is displayed before the training starts

        gpr.train(train_x=train_x, train_y=train_y)
        if np.any(gpr.model_convergence > 0.1):
            warn(
                'GPRs may have not converged! \n'
                'Try more training iterations (training_iter parameter).'
                'You can check the convergence of the training by self._gpr.plot_loss().', RuntimeWarning)
            sleep(1)
        self._gpr = gpr
        ######
        if not isinstance(blackbox_function.design_space, Bounds):
            raise ValueError('Design space property of blackbox function must be an instance of Bounds!')

        if len(pareto_reflections) > 1:
            pareto_reflection = pareto_reflections[1]
            for reflection in pareto_reflections[2:]:
                pareto_reflection = ComposeReflections(reflection, pareto_reflection)
            fun = lambda x: pareto_reflection(gpr(x))
        else:
            fun = lambda x: gpr(x)

        print('\nCalculating optimal scaling...')
        #######
        # Scale g and each component to [0,1]
        epsilon = 2e-2  # smaller epsilon have empirically shown to lead to instabilities

        pareto_reflections[0].scaling_g = calculate_optimal_scaling_g(fun, pareto_reflections[0].g, blackbox_function,
                                                                      self._max_iter_minimizer)
        pareto_reflections[0].scaling_x = calculate_optimal_scaling_x(fun, blackbox_function, self._max_iter_minimizer)
        pareto_reflections[0]._epsilon = epsilon

        ######
        # Optimization
        print('\nOptimization...')

        res = self._minimizer(
            function=lambda x: pareto_reflections[0](fun(x)),
            max_iter=self._max_iter_minimizer,
            upper_bounds=blackbox_function.design_space.upper_bounds,
            lower_bounds=blackbox_function.design_space.lower_bounds,
        )

        print(
            f'\n Found Pareto point: \n x={res} '
            f'\n prediction={gpr(res)} '
            f'\n standard deviation={gpr.std(res)}')
        if np.any([np.all(dominated) for dominated in (gpr(res) >= gpr(blackbox_function.x))]):
            warn(
                'Optimizer did not find a Pareto point! \n'
                'Try more minimizer iterations (max_iter_minimizer).', RuntimeWarning)
            sleep(1)

        # if np.all(pareto_reflection(gpr(res)) >= pareto_reflection(blackbox_function.y[0])):
        #    print('\nNo Pareto point was found. Algorithmic search stopped.')

        # if np.any(np.linalg.norm(gpr(res) - np.array(
        #        [gpr(x) for x in blackbox_function.x]), axis=1) <= self._min_distance_to_evaluated_points):
        #    print('\nFound Pareto point is too close to some already evaluated point.')

        print('\nEvaluating blackbox function...')
        blackbox_function(res)
        print('Value of blackbox function: ', base_blackbox_function.y[-1])
        print('Difference to estimation: ', gpr(res) - base_blackbox_function.y[-1], '\n')
        if base_blackbox_function.y[-1] not in base_blackbox_function.pareto_front:
            warn(
                'Found Point is not Pareto optimal! \n'
                'Either the optimization converged or the optimization failed. Check the convergence by looking at '
                'the difference between the last evaluations (blackbox_function.y).'
                'You can check the convergence of the training by self._gpr.plot_loss().', RuntimeWarning)
            sleep(1)

    @property
    def supported_codomain_dimensions(self) -> None:
        return None
