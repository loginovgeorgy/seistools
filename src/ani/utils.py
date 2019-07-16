import numpy as np
from copy import deepcopy
from itertools import product


ANI_AXIS = np.array([0., 0., 1.], dtype=np.float32)
ANGLE = 0
ROTATION_AXIS = 1


def rotation_matrix(phi, axis, radians=False):
    """
    Build rotation matrix according to given angle phi and axis
    :param phi: angle in degrees (if radians=False) or in radians (if radians=True)
    :param axis: axis along to perform rotation
    :param radians: True/False if False - preform transforms
    :return:
    """
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


def idx_from_3x3_to_6x6(i, j):
    return i * (i == j) + (6 - i - j) * (i != j)


def rotate_cijkl(c_ijkl, a, tol=1e-15):
    """
    Perform rotation of given c_ijkl tensor

    :param c_ijkl: stiffness tensor with shape (3,3,3,3)
    :param a: rotation matrix with shape of (3,3)
    :param tol: tolerance
    :return: rotated c_ijkl
    """
    c_ijkl = deepcopy(c_ijkl)
    a = np.array(deepcopy(a), dtype=np.float32)

    # Is this a rotation matrix?
    if np.sometrue(np.abs(a.dot(a.T) - np.eye(3, dtype=np.float32)) > tol):
        raise RuntimeError('Matrix *A* does not describe a rotation.')

    # Rotate
    c = np.einsum('ia,jb,kc,ld,abcd->ijkl', a, a, a, a, c_ijkl)
    return np.asarray(c)


def rotate_cij(c_ij, a):
    """
    Perform rotation of given c_ijkl tensor
    :param c_ij: stiffness tensor with shape (6,6)
    :param a: rotation matrix with shape of (3,3)
    :return: rotated c_ij
    """
    c_ijkl = cij_to_cijkl(c_ij)
    c_ijkl = rotate_cijkl(c_ijkl, a)
    return cijkl_to_cij(c_ijkl)


def cij_to_cijkl(c_ij):
    """
    Conversion of stiffness tensor from shape (6,6) to (3,3,3,3)
    :param c_ij: stiffness tensor with shape (6,6)
    :return: stiffness tensor with shape (3,3,3,3)
    """
    c_ijkl = np.zeros((3, 3, 3, 3))
    for i, j, k, l in product(*[range(3)] * 4):
        p = idx_from_3x3_to_6x6(i, j)
        q = idx_from_3x3_to_6x6(k, l)
        c_ijkl[i, j, k, l] = c_ij[p, q]
    return c_ijkl


def cijkl_to_cij(c_ijkl):
    """
    Conversion of stiffness tensor from shape from (3,3,3,3) to (6,6)
    :param c_ijkl: stiffness tensor with shape (3,3,3,3)
    :return: stiffness tensor with shape (6,6)
    """
    c_ij = np.zeros((6, 6))

    for i, j, k, l in product(*[range(3)] * 4):
        p = idx_from_3x3_to_6x6(i, j)
        q = idx_from_3x3_to_6x6(k, l)
        c_ij[p, q] = c_ijkl[i, j, k, l]
    return c_ij


def thomsen_to_cij(vp_0=2., vs_0=1., epsilon=0., delta=0., gamma=0., rho=1.):
    """
    Calculate  the stiffness tensor according to Ani=[Vp0, Vs0, epsilon, delta, gamma]
    c33 = Ani(1) ^ 2; c11 = c33 * (1 + 2 * Ani(3));
    c55 = Ani(2) ^ 2; c66 = c55 * (1 + 2 * Ani(5));

    :param vp_0: P-wave velocity
    :param vs_0: S-wave velocity
    :param epsilon:
    :param delta:
    :param gamma:
    :param rho: density
    :return: stiffness tensor with shape (6,6)
    """

    c_ij = np.zeros((6, 6))
    c33 = rho * (vp_0 ** 2)
    c11 = c33 * (1 + 2 * epsilon)
    c55 = rho * (vs_0 ** 2)
    c66 = c55 * (1 + 2 * gamma)

    cc = (c33 - c55) * (2 * delta * c33 + c33 - c55)

    if cc > 0:
        c13 = np.sqrt(cc) - c55
    else:
        c13 = 0

    # Construct Cij
    c_ij[0, 0] = c11
    c_ij[0, 1] = c11 - 2 * c66
    c_ij[0, 2] = c13
    c_ij[1, 1] = c11
    c_ij[1, 2] = c13
    c_ij[2, 2] = c33
    c_ij[3, 3] = c55
    c_ij[4, 4] = c55
    c_ij[5, 5] = c66

    c_ij = reflect_triangle_matrix(c_ij, upper=True)

    return c_ij


def thomsen_to_cij_gpn(vp_0=2., vs_0=1., epsilon=0., delta=0., gamma=0., rho=1.):
    # % % Calculate  the stiffnesses
    # [Vp0, Vs0, epsilon, delta, gamma]
    # c33 = Ani(1) ^ 2; c11 = c33 * (1 + 2 * Ani(3));
    # c55 = Ani(2) ^ 2; c66 = c55 * (1 + 2 * Ani(5));

    c_ij = np.zeros((6, 6))

    c22 = rho * (vp_0 ** 2)
    c33 = rho * (vp_0 ** 2)
    c11 = c33 * (1 + 2 * epsilon)
    c55 = rho * (vs_0 ** 2)
    c66 = rho * (vs_0 ** 2)
    c44 = c55 * (1 + 2 * gamma)
    c23 = c33 - 2 * c44

    cc = (c33 - c55) * (c33 * (1 + 2 * delta) - c55)

    if cc > 0:
        cc = np.sqrt(cc) - c55
    else:
        cc = 0

    c12 = cc
    c13 = cc

    # Construct Cij
    c_ij[0, 0] = c11
    c_ij[1, 1] = c22
    c_ij[2, 2] = c33
    c_ij[3, 3] = c44
    c_ij[4, 4] = c55
    c_ij[5, 5] = c66
    c_ij[0, 1] = c12
    c_ij[0, 2] = c13
    c_ij[1, 2] = c23

    c_ij = reflect_triangle_matrix(c_ij, upper=True)

    return c_ij
