import numpy as np
from .radiation_pattern import radiation_operator


def inverse_operator(x, lam=0):
    e = lam * np.eye(6, 6)
    e[0, 0] = 0
    return np.linalg.pinv(x.T.dot(x) + e)


def moment_inversion(x, d, lam=0):
    xi = inverse_operator(x, lam=lam)
    return xi.dot(x.T).dot(d)


def inverse_data(sou, rec, u_p, u_s, lam=0):
    gp, gs = radiation_operator(sou, rec, notation=6)
    x = np.row_stack([gp, gs])
    if len(u_p.shape) == 3:
        u_p = u_p.reshape(u_p.shape[0], -1)
        u_s = u_s.reshape(u_s.shape[0], -1)
        d = np.hstack([u_p, u_s]).T
    elif len(u_p.shape) == 2:
        d = np.hstack([u_p.ravel(), u_s.ravel()])

    return moment_inversion(x, d, lam=lam)
