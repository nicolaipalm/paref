class ExpressSearch:
    # Purpose: most intuitive but not flexible Pareto front search

    def __init__(self, blackbox_function, constraints):
        self._bbf = blackbox_function
        self._constraints = constraints

        # constraints
    def minimal_search(self):
        # minima components and maximal pareto point
        pass

    def maximal_search(self):
        # edge points and close gaps ie all pareto points
        pass

    def priority_search(self):
        # give percentage for priorities of components
        pass

    def minimize_g(self):
        # minimize some function g
        pass

