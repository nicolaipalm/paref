import numpy as np
from scipy import stats

from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.moo_algorithms.minimizer.surrogates.gpr import GPR


def ehvi_2d(PF, r, mu, sigma):
    # TBA: does not work properly, i.e. does not depend on reference point as it should: ref point too small does not
    #  make EHVI = 0!
    n = PF.shape[0]
    S1 = np.array([r[0], -np.inf])
    S1 = S1.reshape(1, -1)
    Send = np.array([-np.inf, r[1]])
    Send = Send.reshape(1, -1)
    index = np.argsort(PF[:, 1])

    S = PF[index, :]

    S = np.concatenate((S1, S, Send), axis=0)

    y1 = S[:, 0]
    y2 = S[:, 1]

    y1 = y1.reshape(-1, 1)
    y2 = y2.reshape(-1, 1)

    mu = mu.reshape(1, -1)
    sigma = sigma.reshape(1, -1)

    sum_total1 = 0
    sum_total2 = 0

    for i in range(1, n + 2):
        t = (y1[i] - mu[0][0]) / sigma[0][0]
        if i == n + 1:
            sum_total1 = sum_total1
        else:
            sum_total1 = sum_total1 + (y1[i - 1] - y1[i]) * stats.norm.cdf(t) * psi_cal(
                y2[i], y2[i], mu[0][1], sigma[0][1]
            )
        sum_total2 = sum_total2 + (
                psi_cal(y1[i - 1], y1[i - 1], mu[0][0], sigma[0][0])
                - psi_cal(y1[i - 1], y1[i], mu[0][0], sigma[0][0])
        ) * psi_cal(y2[i], y2[i], mu[0][1], sigma[0][1])

    EHVI = sum_total1 + sum_total2
    return EHVI


def psi_cal(a, b, m, s):
    t = (b - m) / s
    return s * stats.norm.pdf(t) + (a - m) * stats.norm.cdf(t)


# print(ehvi_2d(PF=np.array([[0,0]]),r=np.array([0.1,0.1]),mu=np.array([1,1]),sigma=np.array([[0.1],[0.1]])))


class ExpectedHypervolume2d(ParetoReflection):
    def __init__(self, reference_point: np.ndarray, pareto_front: np.ndarray, gpr: GPR):
        self.pareto_front = pareto_front
        self.reference_point = reference_point
        self.gpr = gpr

    def __call__(self, x: np.ndarray):
        return -ehvi_2d(self.pareto_front, self.reference_point, self.gpr(x), self.gpr.std(x))
