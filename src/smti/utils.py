import numpy as np
from scipy.interpolate import griddata


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


def from3x3to6(m):
    return np.array([m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1]])


def from6to3x3(m):
    return np.array([
        [m[0], m[3], m[4]],
        [m[3], m[1], m[5]],
        [m[4], m[5], m[2]],
    ])


def init_equidistant_sphere(n=256):
    """

    :param n:
    :return:
    """

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1/n, 1/n - 1, n)
    radius = np.sqrt(1 - z**2)

    return np.array(
        [
            radius * np.cos(theta),
            radius * np.sin(theta),
            z,
        ]
        , dtype=np.float32
    )


def focal_projection(m, dx=.02, return_grid=False):
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
    if return_grid:
        return u, vij1, vij2, vij3
    return u


def pol2cart(th, r):
    x = r * np.cos(th)
    y = r * np.sin(th)
    return x, y


def projection(plunge):
    rho = np.sin(.5 * np.pi - plunge / 2) / np.sin(np.pi - plunge / 2)
    idx = plunge < .5 * np.pi

    rho[idx] = np.sin(plunge[idx] / 2) / np.sin(.5 * np.pi + plunge[idx] / 2)

    return rho


def interpolate_grid(x, y, z, grid_size=100, method='linear'):
    x = np.float32(x)
    y = np.float32(y)
    z = np.float32(z)
    xg, yg = np.meshgrid(
        np.linspace(x.min(), x.max(), grid_size),
        np.linspace(y.min(), y.max(), grid_size),
    )
    zg = griddata((x, y), z, (xg, yg), method='linear')

    return xg, yg, zg
