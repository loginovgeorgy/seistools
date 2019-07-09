import numpy as np


def projection(m):
    (_, sigma, _) = np.linalg.svd(m)
    sigma = np.sort(sigma)[-1::-1]
    m1, m2, m3 = sigma

    scale = 0.5 * (np.abs(m1 + m2 + m3) + m1 - m3)

    k0 = .5 / scale
    k_iso = k0 * (m1 + m2 + m3)
    k_clvd = k0 * (m1 - 2 * m2 + m3)
    k_dc = k0 * (m1 - m2 - np.abs(m1 + m3 - 2 * m2))

    return k_iso, k_clvd, k_dc


def _alternative_decomposition(m1, m2, m3):
    scale = 0.5 * (np.abs(m1 + m2 + m3) + m1 - m3)
    k0 = 1 / (2 * scale)
    k_iso = k0 * (m1 + m2 + m3)
    k_clvd = k0 * (m1 - 2 * m2 + m3)
    k_dc = k0 * (m1 - m3 - np.abs(m1 + m3 - 2 * m2))
    return k_iso, k_clvd, k_dc


def _basic_decomposition(m1, m2, m3):
    m_iso = (m1 + m2 + m3) / 3
    m_clvd = (2/3) * (m1 + m3 - 2*m2)
    m_dc = .5 * (m1 - m3 - np.abs(m1 + m3 - 2*m2))
    scale = np.abs(m_iso) + np.abs(m_clvd) + m_dc
    k_iso, k_clvd, k_dc = np.array([m_iso, m_clvd, m_dc]) / scale
    return k_iso, k_clvd, k_dc


def decomposition(m, basic=True):
    (sigma, _) = np.linalg.eig(m)

    sigma = np.sort(sigma)[-1::-1]
    sigma = sigma / np.max(np.abs(sigma))
    m1, m2, m3 = sigma

    if basic:
        k_iso, k_clvd, k_dc = _basic_decomposition(m1, m2, m3)
    else:
        k_iso, k_clvd, k_dc = _alternative_decomposition(m1, m2, m3)

    return k_iso, k_clvd, k_dc


def decompose_moment_tensor(m):
    # by Vavricuk

    eig_val, eig_vector = np.linalg.eig(m)

    volumetric = eig_val.mean()
    deviatoric = np.eye(3)*eig_val - volumetric*np.eye(3)

    value_dev, _ = np.linalg.eig(deviatoric)
    abs_values = np.abs(value_dev)
    sort_idx = abs_values.argsort()[::-1]
    value_dev = value_dev[sort_idx]

    maximum_value = value_dev[0]
    minimum_value = value_dev[2]

    max_value_m = np.max(np.abs(eig_val))

    iso = 100 * volumetric / max_value_m
    epsilon = - minimum_value / maximum_value

    clvd = 2 * np.sign(maximum_value) * epsilon * (100 - np.abs(iso))
    dc = 100 - np.abs(iso) - np.abs(clvd)

    return iso, dc, clvd


def decomposition_vu(m):
    """

    :param m:
    :return:
    """
    (sigma, _) = np.linalg.eig(m)

    sigma = np.sort(sigma)[-1::-1]
    scale = np.max(np.abs(sigma))
    sigma = sigma / scale
    m1, m2, m3 = sigma

    u = -(2 / 3) * (m1 + m3 - 2 * m2)
    v = (1 / 3) * (m1 + m2 + m3)

    return v, u

# TODO develop procedure to restore dip, strike, rake
