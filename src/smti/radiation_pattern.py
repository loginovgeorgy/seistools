import numpy as np
from .utils import moment_convert


def _full_to_voigt(g):
    return np.column_stack([
        g[:, 0],
        g[:, 4],
        g[:, 8],
        g[:, 1] + g[:, 3],
        g[:, 2] + g[:, 6],
        g[:, 5] + g[:, 7],
        ])


def radiation_operator(sou, rec, notation=9):
    unit_vec = np.array(rec - sou, dtype=np.float32, ndmin=2)
    # unit_vec *= ((unit_vec ** 2).sum(axis=1, keepdims=True) + 1e-17) ** (-2)
    dist = np.sqrt((unit_vec ** 2).sum(axis=1, keepdims=True)) + 1e-17
    unit_vec /= dist

    gp = np.einsum('pi, pk, pl, p -> pikl', unit_vec, unit_vec, unit_vec, 1 / np.squeeze(dist))
    gs_right = np.einsum('ik, pl, p -> pikl', np.eye(3, 3), unit_vec, 1 / np.squeeze(dist))
    gs = gs_right - gp
    if notation == 9:
        return gp.reshape(-1, 9), gs.reshape(-1, 9)

    if notation == 6:
        return _full_to_voigt(gp.reshape(-1, 9)), _full_to_voigt(gs.reshape(-1, 9))


def calculate_wave_amplitude(sou, rec, m, vp, vs, rho, reshape=False):
    notation = len(m.ravel())

    gp, gs = radiation_operator(sou, rec, notation=notation)
    gp = gp / (4 * np.pi * rho * (vp ** 3))
    gs = gs / (4 * np.pi * rho * (vs ** 3))

    if reshape:
        return gp.dot(m).reshape(-1, 3), gs.dot(m).reshape(-1, 3)

    return gp.dot(m), gs.dot(m)
