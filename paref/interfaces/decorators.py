from typing import List


def initialize_empty_evaluations(func):
    def wrapper(*args, **kwargs):
        """Initialize storage for evaluations of the blackbox function
        """
        args[0]._evaluations = []
        func(*args, **kwargs)

    return wrapper


def store_evaluation_bbf(func):
    def wrapper(*args, **kwargs):
        """Store evaluation of the blackbox function
        """
        result = func(*args, **kwargs)
        args[0]._evaluations.append([args[1], result])
        return result

    return wrapper


def initialize_empty_list_of_pareto_reflections(func):
    def wrapper(*args, **kwargs):
        """Initialize storage for Pareto reflections
        """
        args[0]._used_pareto_reflections = []
        func(*args, **kwargs)

    return wrapper


def store_pareto_reflections(func):
    def wrapper(*args, **kwargs):
        """Store Pareto reflections
        """
        result = func(*args, **kwargs)
        args[0]._used_pareto_reflections.append(result)
        return result

    return wrapper


def store(func, container: List):
    def wrapper(*args, **kwargs):
        """Store output of func in container
        """
        result = func(*args, **kwargs)
        container.append(result)
        func(*args, **kwargs)

    return wrapper
