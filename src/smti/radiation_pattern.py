import numpy as np
from .utils import moment_convert


def radiation_operator(sou, rec):
    unit_vec = np.array(rec - sou, dtype=np.float32, ndmin=2)
    unit_vec *= ((unit_vec ** 2).sum(axis=1, keepdims=True)) ** (-2)

    gp = np.einsum('pi, pk, pl -> pikl', *[unit_vec] * 3)
    gs_right = np.einsum('ik, pl -> pikl', np.eye(3, 3), unit_vec)
    gs = gs_right - gp

    return gp.reshape(-1, 9), gs.reshape(-1, 9)


def calculate_wave_amplitude(sou, rec, m, vp, vs, rho):

    gp, gs = radiation_operator(sou, rec)
    gp = gp / (rho * (vp ** 3))
    gs = gs / (rho * (vs ** 3))

    return gp.dot(m), gs.dot(m)
