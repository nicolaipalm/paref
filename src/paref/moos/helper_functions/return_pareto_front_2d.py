import numpy as np


def return_pareto_front(array: np.array):
    pareto_points = []
    for i, point in enumerate(array):
        is_pareto = True
        for j, other in enumerate(array):
            if i == j:
                continue
            if np.all(point >= other) and np.any(point > other):
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append(point)
    return np.array(pareto_points)


