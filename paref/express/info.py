import numpy as np
import scipy as sp
from scipy.stats import qmc

from paref.blackbox_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.moo_algorithms.minimizer.differential_evolution_minimizer import DifferentialEvolutionMinimizer
from paref.moo_algorithms.minimizer.surrogates.gpr import GPR
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflection_sequences.multi_dimensional.find_1_pareto_points_for_all_components_sequence import \
    Find1ParetoPointsForAllComponentsSequence
from paref.pareto_reflection_sequences.multi_dimensional.find_edge_points_sequence import FindEdgePointsSequence
from paref.pareto_reflections.find_maximal_pareto_point import FindMaximalParetoPoint
from paref.pareto_reflections.minimize_weighted_norm_to_utopia import MinimizeWeightedNormToUtopia

from tabulate import tabulate


class GprBbf(BlackboxFunction):
    """Approximate a blackbox function with a Gaussian process regression (GPR) in the BlackboxFunction interface

    """

    def __init__(self, bbf: BlackboxFunction, training_iter=2000, learning_rate=0.05):
        self._bbf = bbf
        self._evaluations = bbf.evaluations.copy()
        self._gpr = GPR(training_iter=training_iter, learning_rate=learning_rate)
        self._gpr.train(train_x=bbf.x, train_y=bbf.y)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._gpr(x)

    @property
    def dimension_design_space(self) -> int:
        return self._bbf.dimension_design_space

    @property
    def dimension_target_space(self) -> int:
        return self._bbf.dimension_target_space

    @property
    def design_space(self):
        return self._bbf.design_space


class Info:
    """Obtain relevant information about the Pareto front of a blackbox function and model fitness

        * topology:
            the shape of your Pareto front (Info.topology)
        * suggestion_pareto_points:
            suggestions for Pareto points to evaluate, how and why (Info.suggestion_pareto_points)
        * minima:
            the estimated minima of each component (Info.minima)
        * model fitness:
            how well the model approximates the bbf, how to improve it and
            how certain its estimation is (Info.model_fitness)


    .. warning::

        Paref's Info class is still under development.
        If you run into any problems, errors or have suggestions how to make it *user-friendlier*, please contact me
        or open an issue on GitHub. Many thanks!

    """

    def __init__(self, blackbox_function: BlackboxFunction, training_iter=2000, learning_rate=0.05):
        """

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            underlying blackbox function
        training_iter : int default 2000
            number of training iterations for the underlying approximate of the bbf (GPR)
        learning_rate : float default 0.01
            learning rate for the underlying approximate of the bbf (GPR)
        """
        if not isinstance(blackbox_function.design_space, Bounds):
            raise ValueError('Design space property of blackbox function must be an instance of Bounds!')
        self._blackbox_function = blackbox_function
        self._minimizer = DifferentialEvolutionMinimizer()
        self._training_iter = training_iter
        self._learning_rate = learning_rate
        self.update()

    def update(self):
        """Update the information about the Pareto front

        Call this method whenever you have evaluated the blackbox function at some design.
        """
        if len(self._blackbox_function.y) == 0:
            raise ValueError('You must evaluate the blackbox function at least once before obtaining information!')

        print('Obtaining information about the approximate Pareto front...')
        self._surrogate = GprBbf(self._blackbox_function, self._training_iter, self._learning_rate)

        # mean of std
        self.percent_mean_std = np.mean([self._surrogate._gpr.std(x) / self._surrogate._gpr(x) * 100 for x in qmc.scale(
            qmc.LatinHypercube(d=self._blackbox_function.dimension_design_space).random(
                n=1000),
            self._blackbox_function.design_space.lower_bounds,
            self._blackbox_function.design_space.upper_bounds,
        )], axis=0)

        # Search for minima in components
        minima_sequence = Find1ParetoPointsForAllComponentsSequence()

        self._minimizer.apply_to_sequence(blackbox_function=self._surrogate,
                                          sequence_pareto_reflections=minima_sequence,
                                          stopping_criteria=MaxIterationsReached(
                                              max_iterations=self._blackbox_function.dimension_target_space))
        self._minima_pareto_points = minima_sequence.best_fits(self._surrogate.y)
        self._minima_pareto_points_std = [self._surrogate._gpr.std(x) for i, x in
                                          enumerate(
                                              self._surrogate.x[-self._blackbox_function.dimension_target_space:])]
        self._minima = np.min(minima_sequence.best_fits(self._surrogate.y), axis=0)
        self._minima_std = [self._surrogate._gpr.std(x)[i] for i, x in
                            enumerate(self._surrogate.x[-self._blackbox_function.dimension_target_space:])]

        # Search for maximal Pareto point
        maximal_pareto_point_reflection = FindMaximalParetoPoint(blackbox_function=self._surrogate)
        self._minimizer.apply_to_sequence(blackbox_function=self._surrogate,
                                          sequence_pareto_reflections=maximal_pareto_point_reflection,
                                          stopping_criteria=MaxIterationsReached(max_iterations=1))
        self.maximal_pareto_point = maximal_pareto_point_reflection.best_fits(self._surrogate.y)[-1]
        self.maximal_pareto_point_std = self._surrogate._gpr.std(self._surrogate.x[-1])

        # Search for closest to theoretical global optimum
        global_optimum_pareto_point_reflection = (
            MinimizeWeightedNormToUtopia(utopia_point=self._minima,
                                         potency=2,
                                         scalar=np.ones(
                                             self._surrogate.dimension_target_space)))
        self._minimizer.apply_to_sequence(blackbox_function=self._surrogate,
                                          sequence_pareto_reflections=global_optimum_pareto_point_reflection,
                                          stopping_criteria=MaxIterationsReached(max_iterations=1))
        self.global_optimum_pareto_point = maximal_pareto_point_reflection.best_fits(self._surrogate.y)[-1]
        self.global_optimum_pareto_point_std = self._surrogate._gpr.std(self._surrogate.x[-1])

        # Search for edge points
        if self._blackbox_function.dimension_target_space > 2:
            edge_points_sequence = FindEdgePointsSequence()
            self._minimizer.apply_to_sequence(blackbox_function=self._surrogate,
                                              sequence_pareto_reflections=edge_points_sequence,
                                              stopping_criteria=MaxIterationsReached(
                                                  max_iterations=self._blackbox_function.dimension_target_space))
            self.edge_points = minima_sequence.best_fits(self._surrogate.y)

        else:
            self.edge_points = self._minima_pareto_points

        # Concave/convex
        basis = sp.linalg.orth((self.edge_points - self.edge_points[0]).T).T

        self.dimension_pf = len(basis)  # approximation of dimension of Pareto front
        projected_point = np.sum(
            np.array([np.dot(self.maximal_pareto_point - self.edge_points[0], basis_vector) * basis_vector
                      for basis_vector in basis]), axis=0) + self.edge_points[0]
        min_edges = np.min(self.edge_points, axis=0)

        # TODO: this is prone to small values!
        self._concave_degree = np.linalg.norm(
            self.maximal_pareto_point - min_edges) / np.linalg.norm(
            projected_point - min_edges)

        if self._concave_degree < 0.5:
            self.concave_degree_description = 'Very convex'

        elif self._concave_degree < 0.97:
            self.concave_degree_description = 'Convex'

        elif self._concave_degree < 1.03:
            self.concave_degree_description = 'Linear'

        elif self._concave_degree < 1.3:
            self.concave_degree_description = 'Concave'

        else:
            self.concave_degree_description = 'Very concave'

        # is there a trade-off
        # TODO: global optimum check might fail: testing on real use-cases
        self.global_optimum = False
        if self.dimension_pf == 0:
            self.global_optimum = True

        print("""Done! You can access the following information about your Pareto front:
        * model fitness: how well the model approximates the bbf,
            how to improve it and how certain its estimation is (Info.model_fitness)
        * topology:
            topological information of your Pareto front (Info.topology)
        * suggestion:
            for Pareto points to evaluate, how and why (Info.suggestion_pareto_points)
        * minima:
            the estimated minima of each component (Info.minima)
        """)

    @property
    def minima(self):
        """Estimated minimum of each component
        """

        print(tabulate({'Component': range(len(self._minima)),
                        'Minimum': self._minima,
                        'Std': self._minima_std}, headers='keys'))

    @property
    def topology(self):
        """Obtain topological information about the Pareto front

        Is there are global optimum?
        Is the Pareto front rather convex or concave or linear?
        Of which dimension is the Pareto front?
        """

        print(
            tabulate({'(Almost) Global optimum': [self.global_optimum],
                      'Shape': [self.concave_degree_description],
                      'Dimension Pareto front': [self.dimension_pf]},
                     headers='keys'))

    @property
    def suggestion_pareto_points(self):
        """Suggestions for Pareto points to evaluate, how and why

        The suggestion mainly focuses on the approximate shape of the Pareto front and its implication
        for the trade-offs between the components.
        """
        if self.global_optimum:
            print("""
            There is probably an (almost) global optimum.
            This means your target objectives are not conflicting
            and any Pareto point will be that global optimum.
            You can use any MOO algorithm provided in Paref to find that point.
            """)

        elif self._concave_degree < 0.7:
            print("""Your objectives appear to be conflicting, so there are real trade-offs.
            Since your Pareto front appears to be highly convex, you may be able to achieve large improvements
            in one component while suffering relatively small losses in others.
            The best trade-off I've found that minimises all components simultaneously is
            """)
            print(tabulate({'Target values': [self.global_optimum_pareto_point.tolist()],
                            'Std': [self.global_optimum_pareto_point_std.tolist()],
                            'Dominates x% of evaluations': [
                                self._better_than_evaluations(self.global_optimum_pareto_point)]},
                           headers='keys'))
            print(f"""This Pareto optimum is closest to the theoretical global optimum:
            {self._minima}
            """)
            print("""
            You can use the search_for_best_real_trade_off algorithm provided by the paref.express.express_search
            module to find that Pareto point.
            """)

        elif self._concave_degree < 0.97:
            print("""
            Your objectives appear to be conflicting, so there are real trade-offs.
            As your Pareto front appears to be convex, you may be able to achieve improvements in one
            component while suffering a smaller loss in others.
            A Pareto point that represents a real trade-off in all components I've found is
            """)
            print(tabulate({'Target values': [self.maximal_pareto_point.tolist()],
                            'Std': [self.maximal_pareto_point_std.tolist()],
                            'Dominates x% of evaluations': [self._better_than_evaluations(self.maximal_pareto_point)]},
                           headers='keys'))
            print("""
            You can use the search_for_best_real_trade_off algorithm provided by the paref.express.express_search
            module to find that Pareto point.
            """)

        elif self._concave_degree < 1.03:
            print("""
            Your objectives appear to be conflicting, so there are real trade-offs.
            Your Pareto front appears to be (almost) a plane.
            This means that improvements in one component will result in
            almost equal losses in other components.
            Accordingly, all Pareto points will be (almost) equally good and it is up to you to
            decide which components are more relevant.
            I suggest you use the priority_search algorithm provided by the paref.express.express_search module
            to take into account your preference for certain components.
            """)

        elif self._concave_degree < 1.3:
            print("""
            Your objectives appear to be conflicting, so there are real trade-offs.
            Your Pareto front appears to be concave.
            This means that any improvement in one component will result in a greater loss in the other components.
            Accordingly, I suggest that you focus on a single component by using
            the search_for_minima or priority_search algorithm provided by the paref.express.express_search module
            to take into account your preference for certain components.
            Here are the estimated Pareto points that minimise a component
            """)

            print(tabulate(
                [[i, minima.tolist(), self._minima_pareto_points_std[i].tolist(), self._better_than_evaluations(minima)]
                 for i, minima in enumerate(self._minima_pareto_points)],
                headers=['Component', 'Target values', 'Std', 'Dominates x% of evaluations']))

            suggested_point = self._minima_pareto_points[
                np.argmax([self._better_than_evaluations(minima) for minima in self._minima_pareto_points])]
            print(f"""
            I propose
            {suggested_point}
            because it dominates the most evaluations ({self._better_than_evaluations(suggested_point)}%).

            Nevertheless, here is a Pareto point representing a real trade-off in all components
            """)
            print(tabulate({'Target values': [self.maximal_pareto_point.tolist()],
                            'Std': [self.maximal_pareto_point_std.tolist()],
                            'Dominates x% of evaluations': [self._better_than_evaluations(self.maximal_pareto_point)]},
                           headers='keys'))

        else:
            print("""
            Your Pareto front appears to be very concave.
            This means that any little improvement in one component will result in a significant
            loss in the other components.
            Accordingly, I suggest that you focus on a single component and
            choose the Pareto point that minimises that component.
            You can use the search_for_minima algorithm provided by the paref.express.express_search module
            to find those Pareto points.

            Here are the estimated Pareto points that minimise a component
            """)
            print(tabulate([[i,
                             minima.tolist(),
                             self._minima_pareto_points_std[i].tolist(),
                             self._better_than_evaluations(minima)] for i, minima in
                            enumerate(self._minima_pareto_points)
                            ], headers=['Component', 'Target values', 'Std', 'Dominates x% of evaluations']))
            print(f"""
            I propose
            {self._minima_pareto_points[
                np.argmax([self._better_than_evaluations(minima) for minima in self._minima_pareto_points])]}
            because it dominates the most evaluations.
            """)

    @property
    def model_fitness(self):
        """Information about the model fitness

        How well does the model approximate the blackbox function?
        Are the training iterations sufficient, i.e. did the training convergence?
        How uncertain is the model?
        How many evaluations are needed to obtain good results?
        """
        # TODO: include cross validation
        losses = np.array([hp['loss'] for hp in self._surrogate._gpr.info])
        losses_last = np.array([hp['loss'][int(self._training_iter * 0.9):] for hp in self._surrogate._gpr.info])

        model_convergence = (np.max(losses_last, axis=1) - np.min(losses_last, axis=1)) / (
                np.max(losses, axis=1) - np.min(losses, axis=1)) * 9

        # TODO: this might be prone to errors. Try a relative convergence criterion instead.
        if np.all(model_convergence < 0.05):
            print(
                """
                The model has converged. The training iterations seem to be sufficient.
                """
            )
        else:
            print("""
                The model has not converged. I suggest you increase the number of training iterations.
                """)

        if np.any(self.percent_mean_std > 10):
            print("""
                The uncertainty of your model seems to be high.
                Accordingly, you might need more evaluations to obtain good results.
                I suggest you allow each algorithm the maximal number of evaluations
                by using the max_iterations stopping criterion.
                """)
        else:
            print("""
                The uncertainty of your model seems to be low.
                Accordingly, you might only need few evaluations to obtain good results.
                I suggest you allow each algorithm the minimal number of evaluations
                by using the convergence_reached stopping criterion.
                """)

        print(tabulate({'Component': range(self._blackbox_function.dimension_target_space),
                        f'Average uncertainty (%) at {self._training_iter} training iterations':
                            self.percent_mean_std.tolist()},
                       headers='keys'))
        self._surrogate._gpr.plot_loss()

    def _better_than_evaluations(self, y):
        i = 0
        for eval in self._blackbox_function.y:
            if np.all(eval >= y):
                i += 1

        return i / len(self._blackbox_function.y) * 100  # in percent

    @property
    def model(self) -> GPR:
        """The underlying model of the blackbox function

        Returns
        -------
        GPR
            underlying surrogate (GPR) of the blackbox function
        """
        return self._surrogate._gpr
