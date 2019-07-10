import numpy as np
from .utils import moment_convert


def radiation_operator(sou, rec):
    unit_vec = np.array(rec - sou, dtype=np.float32, ndmin=2)
    # unit_vec *= ((unit_vec ** 2).sum(axis=1, keepdims=True) + 1e-17) ** (-2)
    dist = np.sqrt((unit_vec ** 2).sum(axis=1, keepdims=True)) + 1e-17
    unit_vec /= dist

    gp = np.einsum('pi, pk, pl, p -> pikl', unit_vec, unit_vec, unit_vec, 1 / np.squeeze(dist))
    gs_right = np.einsum('ik, pl, p -> pikl', np.eye(3, 3), unit_vec, 1 / np.squeeze(dist))
    gs = gs_right - gp
    # gp = np.einsum('pikl, p -> pikl', gp, 1 / np.squeeze(dist))
    # gs = np.einsum('pikl, p -> pikl', gs, 1 / np.squeeze(dist))
    return gp.reshape(-1, 9), gs.reshape(-1, 9)


def calculate_wave_amplitude(sou, rec, m, vp, vs, rho, reshape=False):

    gp, gs = radiation_operator(sou, rec)
    gp = gp / (4 * np.pi * rho * (vp ** 3))
    gs = gs / (4 * np.pi * rho * (vs ** 3))

    if reshape:
        return gp.dot(m).reshape(-1, 3), gs.dot(m).reshape(-1, 3)

    return gp.dot(m), gs.dot(m)

