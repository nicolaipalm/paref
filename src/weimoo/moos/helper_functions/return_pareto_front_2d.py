import numpy as np


def return_pareto_front_2d(array: np.array):
    PF = []
    for element in array:
        for other_element in array:
            dominated = False
            if other_element[0] <= element[0] and other_element[1] <= element[1] and (
                    other_element[0] < element[0] or other_element[1] < element[1]):
                dominated = True
                break

        if not dominated:
            PF.append(element)

    return np.array(PF)
