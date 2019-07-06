import numpy as np


def moment_convert(m):
    m = np.array(m)
    if len(m.ravel()) == 6:
        m = np.array(
            [[m[0], m[5], m[4]],
             [m[5], m[1], m[3]],
             [m[4], m[3], m[2]], ]
        )
    elif len(m.ravel()) == 9:
        m = np.array(
            [m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1]]
        )

    return m


def focal_projection(m, dx=.02):
    """
    m must be voigt-like notation [m11 m22 m33 m23 m13 m12]

    :param m:
    :param dx:
    :return:
    """
    dy = dx  # grid loop
    x = np.arange(-1, 1, dx)[None, ...]
    y = np.arange(-1, 1, dy)[..., None]

    nx = len(x)
    ny = len(y)  # vectorization of previous code begins here

    x2 = x.repeat(ny, axis=0)
    y2 = y.repeat(ny, axis=1)

    r2 = x2 * x2 + y2 * y2
    trend = np.arctan2(y2, x2)
    plunge = np.pi / 2 - 2 * np.arcsin(np.sqrt(r2 / 2))  # equal area projection

    vij1 = np.cos(trend) * np.cos(plunge)  # set up local vector grids
    vij2 = np.sin(trend) * np.cos(plunge)
    vij3 = np.sin(plunge)

    m = np.array(m)
    if len(m.ravel()) == 9:
        m = moment_convert(m)

    u1 = (vij1 * m[0] + vij2 * m[5] + vij3 * m[4]) * vij1
    u2 = (vij1 * m[5] + vij2 * m[1] + vij3 * m[3]) * vij2
    u3 = (vij1 * m[4] + vij2 * m[3] + vij3 * m[2]) * vij3
    u = u1 + u2 + u3
    u[r2 > 1] = np.nan
    return u, vij1, vij2, vij3