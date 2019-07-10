import numpy as np
from copy import deepcopy


def rotation_matrix(phi, axis, radians=False):
    if not radians:
        phi = phi * np.pi / 180

    a = np.eye(3)
    if axis == 0:
        a = np.array([
            [1, 0, 0],
            [0, np.cos(phi), np.sin(phi)],
            [0, -np.sin(phi), np.cos(phi)]
        ])
    if axis == 1:
        a = np.array([
            [np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [-np.sin(phi), 0, np.cos(phi)]
        ])

    if axis == 2:
        a = np.array([
            [np.cos(phi), np.sin(phi), 0],
            [-np.sin(phi), np.cos(phi), 0],
            [0, 0, 1],
        ])

    return a


def reflect_triangle_matrix(x, upper=True):
    x = deepcopy(x)

    if len(x.shape) != 2:
        raise ValueError('Matrix must be 2D array')
    if x.shape[0] != x.shape[1]:
        raise ValueError('Matrix must be squared')

    # shape = x.shape
    # x = (x + x.T) * (np.ones(shape) - .5 * np.eye(*shape))

    idx_u = np.triu_indices(max(x.shape))
    idx_l = idx_u[::-1]
    if not upper:
        x = x.T

    x[idx_l] = x[idx_u]
    return x


def vector_from_angles(azimuth, polar, r=1, radians=True):
    if not radians:
        azimuth = azimuth * np.pi / 180
        polar = polar * np.pi / 180

    x = r * np.sin(polar) * np.cos(azimuth)
    y = r * np.sin(polar) * np.sin(azimuth)
    z = r * np.cos(polar)

    return np.array([x, y, z], dtype=np.float32)


def angles_from_vector(x, y, z, radians=True):
    r = np.sqrt(x**2 + y**2 + z**2)

    az = np.arctan2(y, x)
    el = np.arctan2(z, r)

    if not radians:
        az = az * 180 / np.pi
        el = el * 180 / np.pi

    return np.array([az, el, r], dtype=np.float32)


def cart2sph(x, y, z, radians=True):
    """
    Convert cartesian to spherical coordinates
    :param x: X / East
    :param y: Y / North
    :param z: Z / Depth
    :param radians: return output in radians
    :return:
    azimuth, elevation, radius
    """
    hxy = np.hypot(x, y)
    r = np.sqrt(x**2 + y**2 + z**2)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)

    if not radians:
        az = az * 180 / np.pi
        el = el * 180 / np.pi

    return np.array([az, el, r], dtype=np.float32)


def sph2cart(az, el, r=1, radians=True):
    """
    Convert spherical coordinates to cartesian
    :param az: Azimuth
    :param el: Elevation
    :param r: Radius
    :param radians: is input in radians
    :return:
    X / East, Y / North, Z / Depth
    """
    if not radians:
        az = az * np.pi / 180
        el = el * np.pi / 180

    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return np.array([x, y, z], dtype=np.float32)


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
