import numpy as np

from paref.pareto_reflections.restrict_by_point import RestrictByPoint


def test_reflecting_example_case():
    nadir, restricting_point = np.array([3, 7]), np.zeros(2)
    pareto_reflection = RestrictByPoint(nadir=nadir, restricting_point=restricting_point)
    assert ((pareto_reflection(np.ones(2)) == nadir).all())


def test_reflecting_dominating_2d_cases():
    nadir, restricting_point = np.array([2, 2]), np.ones(2)
    pareto_reflection = RestrictByPoint(nadir=nadir, restricting_point=restricting_point)

    case = np.zeros(2)
    assert ((pareto_reflection(case) == case).all())

    case = np.array([1, 0.9])  # is dominating in 1 dimension
    assert ((pareto_reflection(case) == case).all())

    case = np.ones(2)  # is not better in at least one dimension (equal case)
    assert ((pareto_reflection(case) == case).all())

    case = np.ones(2) * 0.9
    assert ((pareto_reflection(case) == case).all())

    case = np.ones(2) * 0.9999999999
    assert ((pareto_reflection(case) == case).all())


def test_reflecting_non_dominating_2d_cases():
    nadir, restricting_point = np.array([2, 2]), np.ones(2)
    pareto_reflection = RestrictByPoint(nadir=nadir, restricting_point=restricting_point)

    case = np.ones(2) * 1.1
    assert ((pareto_reflection(case) == nadir).all())

    case = np.array([1.1, 0.9])  # is not at least as good in all dimensions
    assert ((pareto_reflection(case) == nadir).all())


def test_reflecting_dominating_2d_edge_cases():
    nadir, restricting_point = np.array([2, 2]), np.array([2, 2])  # nadir and restricting_point are equal
    pareto_reflection = RestrictByPoint(nadir=nadir, restricting_point=restricting_point)

    case = np.ones(2)  # is better in both dimensions
    assert ((pareto_reflection(case) == case).all())


def test_reflecting_non_dominating_2d_edge_cases():
    nadir, restricting_point = np.array([2, 2]), np.ones(2)
    pareto_reflection = RestrictByPoint(nadir=nadir, restricting_point=restricting_point)

    case = np.array([3, 3])  # is worse than nadir
    assert ((pareto_reflection(case) == nadir).all())

    nadir, restricting_point = np.array([2, 2]), np.array([2, 2])  # nadir and restricting_point are equal
    pareto_reflection = RestrictByPoint(nadir=nadir, restricting_point=restricting_point)

    assert ((pareto_reflection(case) == nadir).all())


def test_reflecting_dominating_3d_cases():
    nadir, restricting_point = np.array([2, 2, 2]), np.ones(3)
    pareto_reflection = RestrictByPoint(nadir=nadir, restricting_point=restricting_point)

    case = np.zeros(3)
    assert ((pareto_reflection(case) == case).all())

    case = np.array([1, 1, 0.9])  # is dominating in 1 dimension
    assert ((pareto_reflection(case) == case).all())

    case = np.ones(3)  # is not better in at least one dimension (equal case)
    assert ((pareto_reflection(case) == case).all())

    case = np.ones(3) * 0.9
    assert ((pareto_reflection(case) == case).all())

    case = np.ones(3) * 0.9999999999
    assert ((pareto_reflection(case) == case).all())


def test_reflecting_non_dominating_3d_cases():
    nadir, restricting_point = np.array([2, 2, 2]), np.ones(3)
    pareto_reflection = RestrictByPoint(nadir=nadir, restricting_point=restricting_point)

    case = np.ones(3) * 1.1
    assert ((pareto_reflection(case) == nadir).all())

    case = np.array([1.1, 0.9, 0.9])  # is not at least as good in all dimensions
    assert ((pareto_reflection(case) == nadir).all())


def test_reflecting_dominating_3d_edge_cases():
    nadir, restricting_point = np.array([3, 3, 3]), np.array([3, 3, 3])  # nadir and restricting_point are equal
    pareto_reflection = RestrictByPoint(nadir=nadir, restricting_point=restricting_point)

    case = np.ones(3)  # is better in both dimensions
    assert ((pareto_reflection(case) == case).all())


def test_reflecting_non_dominating_3d_edge_cases():
    nadir, restricting_point = np.array([3, 3, 3]), np.ones(3)
    pareto_reflection = RestrictByPoint(nadir=nadir, restricting_point=restricting_point)

    case = np.array([4, 4, 4])  # is worse than nadir
    assert ((pareto_reflection(case) == nadir).all())

    nadir, restricting_point = np.array([3, 3, 3]), np.array([3, 3, 3])  # nadir and restricting_point are equal
    pareto_reflection = RestrictByPoint(nadir=nadir, restricting_point=restricting_point)

    assert ((pareto_reflection(case) == nadir).all())
