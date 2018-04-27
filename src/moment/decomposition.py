import numpy as np


def _alternative_decomposition(m1, m2, m3):
    """

    :param m1:
    :param m2:
    :param m3:
    :return:
    """
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


def decomposition(m, dec_type='Alternative'):
    (sigma, _) = np.linalg.eig(m)

    sigma = np.sort(sigma)[-1::-1]
    sigma = sigma / np.max(np.abs(sigma))
    m1, m2, m3 = sigma

    k_iso, k_clvd, k_dc = _alternative_decomposition(m1, m2, m3)

    return k_iso, k_clvd, k_dc


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


def projection(M):
    (_, sigma, _) = np.linalg.svd(M)
    sigma = np.sort(sigma)[-1::-1]
    m1, m2, m3 = sigma

    scale = 0.5 * (np.abs(m1 + m2 + m3) + m1 - m3)

    k0 = .5 / scale
    k_iso = k0 * (m1 + m2 + m3)
    k_clvd = k0 * (m1 - 2 * m2 + m3)
    k_dc = k0 * (m1 - m2 - np.abs(m1 + m3 - 2 * m2))

    return k_iso, k_clvd, k_dc
