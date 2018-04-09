import numpy as np


def _opening_moment(vec, dn, n, ng):
    """

    :param vec:
    :param dn:
    :param n:
    :param ng:
    :return:
    """

    p, d, l = vec
    opening = np.array([dn, -dn, dn, dn, -dn, dn])
    opening_vec = np.array([
        (ng + 2 * n * (np.sin(p) ** 2) * (np.sin(d) ** 2)),
        n * np.sin(2 * p) * (np.sin(d) ** 2),
        n * np.sin(p) * np.sin(2 * d),
        (ng + 2 * n * (np.cos(p) ** 2) * (np.sin(d) ** 2)),
        n * np.cos(p) * np.sin(2 * d),
        (ng + 2 * n * (np.cos(d) ** 2))
    ])

    return opening * opening_vec


def _shear_moment(vec, ds, n, ng):
    """

    :param vec:
    :param ds:
    :param n:
    :param ng:
    :return:
    """
    from numpy import sin, cos
    p, d, l = vec
    shear = np.array([-ds, ds, -ds, ds, -ds, ds])
    shear_vec = n * np.array([
        (sin(2 * p) * sin(d) * cos(l) + (sin(p) ** 2) * sin(2 * d) * sin(l)),
        (cos(2 * p) * sin(d) * cos(l) + 0.5 * sin(2 * p) * sin(2 * d) * sin(l)),
        (cos(p) * cos(d) * cos(l) + sin(p) * cos(2 * d) * sin(l)),
        (sin(2 * p) * sin(d) * cos(l) - (cos(p) ** 2) * sin(2 * d) * sin(l)),
        (sin(p) * cos(d) * cos(l) - cos(p) * cos(2 * d) * sin(l)),
        sin(2 * d) * sin(l)
    ])
    return shear * shear_vec


def _general_moment_tensor(vec, ds=1, dn=0, n=1, ng=1.4, a=1):
    """

    :param vec:
    :param ds:
    :param dn:
    :param n:
    :param ng:
    :param a:
    :return:
    """
    m = _shear_moment(vec, ds, n, ng) + _opening_moment(vec, dn, n, ng)
    m = a * m
    return m


def _general_double_couple(v):
    """

    :param v:
    :return: moment tensor vector m = [xx, xy, xz, yy, yz, zz]
    """
    l, d, p = v

    xx = -(np.sin(d) * np.cos(l) * np.sin(2 * p) + np.sin(2 * d) * np.sin(l) * (np.sin(p) ** 2))
    yy = (np.sin(d) * np.cos(l) * np.sin(2 * p) - np.sin(2 * d) * np.sin(l) * (np.cos(p) ** 2))
    zz = np.sin(2 * d) * np.sin(l)
    xy = (np.sin(d) * np.cos(l) * np.cos(2 * p) + .5 * np.sin(2 * d) * np.sin(l) * np.sin(2 * p))
    xz = -(np.cos(d) * np.cos(l) * np.cos(p) + np.cos(2 * d) * np.sin(l) * np.sin(p))
    yz = -(np.cos(d) * np.cos(l) * np.sin(p) - np.cos(2 * d) * np.sin(l) * np.cos(p))

    m = np.array([xx, xy, xz, yy, yz, zz])
    m = np.float16(m)
    m = np.float32(m)
    #     m[abs(m)<4*eps] = 0
    return m

