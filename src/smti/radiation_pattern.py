import numpy as np


def radiation_operator(sou, rec, axis=1):

    # for notation [m11, m22, m33, m23, m13, m12]
    idx_k = np.array([1, 2, 3, 2, 1, 1]) - 1
    idx_l = np.array([1, 2, 3, 3, 3, 2]) - 1

    # for notation [m11, m12, m13, m22, m23, m33]
    idx_k = np.array([1, 1, 1, 2, 2, 3]) - 1
    idx_l = np.array([1, 2, 3, 2, 3, 3]) - 1

    factor = np.array([1, 1, 1, 2, 2, 2])

    dim = sou.shape[1]
    unit_vec = np.array(rec - sou, dtype=np.float32, ndmin=2)
    norm = (4 * np.pi (unit_vec ** 2).sum(axis=0)) ** (-1)

    unit_vec_k = unit_vec[:, idx_k]
    unit_vec_l = unit_vec[:, idx_k]

    gp = np.einsum('pi, pl, pl, p, l -> pil', unit_vec, unit_vec_k, unit_vec_l, norm, factor)
    gs_left = np.einsum('pi, pl, pl, l -> pil', unit_vec, unit_vec_k, unit_vec_l)
    gs_right = np.einsum('ik, pl -> pil', np.eye(dim, dim), unit_vec_l)
    gs = - (gs_left - gs_right)
    gs = np.einsum('pil, p -> pil', gs, norm)

    return gp, gs


def calculate_wave_amplitude(sou, rec, m, vp, vs, rho):

    dim = sou.shape[1]
    unit_vec = np.array(rec - sou, dtype=np.float32, ndmin=2)
    norm = (4 * np.pi * rho * (unit_vec ** 2).sum(axis=0)) ** (-1)

    gp = np.einsum('pi, pk, pl, p, kl -> pi', unit_vec, unit_vec, unit_vec, norm, m)
    gp = gp / (vp ** 3)

    gs_left = np.einsum('pi, pk, pl, kl -> pi', unit_vec, unit_vec, unit_vec, m)
    gs_right = np.einsum('ik, pl, kl -> pi', np.eye(dim, dim), unit_vec, m)
    gs = - (gs_left - gs_right)
    gs = np.einsum('pi, p -> pi', gs, norm)
    gs = gs / (vs ** 3)

    return gp, gs
