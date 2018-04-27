import numpy as np


def cart2sph(x, y, z):
    """
    Convert cartesian to spherical coordinates
    :param x: X / East
    :param y: Y / North
    :param z: Z / Depth
    :return:
    azimuth, elevation, radius
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def sph2cart(az, el, r):
    """
    Convert spherical coordinates to cartesian
    :param az: Azimuth
    :param el: Elevation
    :param r: Radius
    :return:
    X / East, Y / North, Z / Depth
    """
    r_cos_theta = r * np.cos(el)
    x = r_cos_theta * np.cos(az)
    y = r_cos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z