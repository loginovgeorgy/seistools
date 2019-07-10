import numpy as np
from copy import deepcopy
from itertools import product
from .utils import sph2cart, cart2sph, reflect_triangle_matrix, rotation_matrix, init_equidistant_sphere

ANI_AXIS = np.array([0., 0., 1.], dtype=np.float32)
ANGLE = 0
ROTATION_AXIS = 1


def idx_from_3x3_to_6x6(i, j):
    return i * (i == j) + (6 - i - j) * (i != j)


def rotate_cijkl(c_ijkl, a, tol=1e-6):
    """

    :param c_ijkl:
    :param a:
    :param tol:
    :return:
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
    c_ijkl = cij_to_cijkl(c_ij)
    c_ijkl = rotate_cijkl(c_ijkl, a)
    return cijkl_to_cij(c_ijkl)


# def rotate_cijkl(cijkl, a):
#     cijkl = cijkl.copy()
#
#     c = np.zeros((3, 3, 3, 3))
#     for i, j, k, l, i1, j1, k1, l1 in product(*[range(3)] * 8):
#         tmp = a[i, i1] * a[j, j1] * a[k, k1] * a[l, l1]
#         c[i, j, k, l] += tmp * cijkl[i1, j1, k1, l1]
#     return c


def cij_to_cijkl(c_ij):
    c_ijkl = np.zeros((3, 3, 3, 3))
    for i, j, k, l in product(*[range(3)] * 4):
        p = idx_from_3x3_to_6x6(i, j)
        q = idx_from_3x3_to_6x6(k, l)
        c_ijkl[i, j, k, l] = c_ij[p, q]
    return c_ijkl


def cijkl_to_cij(c_ijkl):
    c_ij = np.zeros((6, 6))

    for i, j, k, l in product(*[range(3)] * 4):
        p = idx_from_3x3_to_6x6(i, j)
        q = idx_from_3x3_to_6x6(k, l)
        c_ij[p, q] = c_ijkl[i, j, k, l]
    return c_ij


def _calculate_christ_tensor(cij, n):
    g_christ = np.zeros((3, 3))
    for i, j, k, l in product(*[range(3)] * 4):
        p = idx_from_3x3_to_6x6(i, j)
        q = idx_from_3x3_to_6x6(k, l)
        g_christ[j, l] += (cij[p, q] * n[i] * n[k])

    return g_christ


def _calculate_phase_and_polarization(g, n):
    _, s, v = np.linalg.svd(g)

    v = np.real(v)
    s = np.real(s)
    s = np.abs(s)
    s = np.sqrt(s)

    idx_sort = np.abs(s).argsort()[::-1]
    v = v[idx_sort]

    if n.dot(v).dot(n) < 0:
        v = -v

    return s, v


def calculate_wave(c_ij, n, ani_axis=None, rho=1, tol=1e-15):
    if not isinstance(ani_axis, np.ndarray):
        ani_axis = np.array([0, 0, 1], dtype=np.float32)

    rho = np.sqrt(rho) + tol

    g_christ = _calculate_christ_tensor(c_ij, n)
    v_phase, polarization = _calculate_phase_and_polarization(g_christ, n)

    v_phase /= rho
    v_phase += tol

    polarization_p = polarization[:, 0].copy()
    v_phase_p = v_phase[0].copy()

    if np.abs(polarization[:, 2].dot(ani_axis)) <= tol:
        polarization_sv = polarization[:, 1].copy()
        polarization_sh = polarization[:, 2].copy()
        v_phase_sv = v_phase[1].copy()
        v_phase_sh = v_phase[2].copy()
    else:
        polarization_sh = polarization[:, 1].copy()
        polarization_sv = polarization[:, 2].copy()
        v_phase_sv = v_phase[2].copy()
        v_phase_sh = v_phase[1].copy()

    ray_p = np.zeros(3)
    ray_sv = np.zeros(3)
    ray_sh = np.zeros(3)

    for i, j, k, l in product(*[range(3)] * 4):
        p = idx_from_3x3_to_6x6(i, j)
        q = idx_from_3x3_to_6x6(k, l)
        ray_p[i] += c_ij[p, q] * n[k] * polarization_p[j] * polarization_p[l]
        ray_sv[i] += c_ij[p, q] * n[k] * polarization_sv[j] * polarization_sv[l]
        ray_sh[i] += c_ij[p, q] * n[k] * polarization_sh[j] * polarization_sh[l]

    ray_p = ray_p / v_phase_p
    ray_sv = ray_sv / v_phase_sv
    ray_sh = ray_sh / v_phase_sh

    azimuth, polar, _ = cart2sph(*n, radians=False)
    azimuth_p, polar_p, v_group_p = cart2sph(*ray_p, radians=False)
    azimuth_sv, polar_sv, v_group_sv = cart2sph(*ray_sv, radians=False)
    azimuth_sh, polar_sh, v_group_sh = cart2sph(*ray_sh, radians=False)

    ng_p = sph2cart(azimuth_p.ravel(), polar_p.ravel(), 1, radians=True)
    ng_sv = sph2cart(azimuth_sv.ravel(), polar_sv.ravel(), 1, radians=True)
    ng_sh = sph2cart(azimuth_sh.ravel(), polar_sh.ravel(), 1, radians=True)

    layer = dict(
        polarization_p=polarization_p,
        polarization_sv=polarization_sv,
        polarization_sh=polarization_sh,
        v_group_p=v_group_p,
        v_group_sv=v_group_sv,
        v_group_sh=v_group_sh,
        v_phase_p=v_phase_p,
        v_phase_sv=v_phase_sv,
        v_phase_sh=v_phase_sh,
        azimuth_p=azimuth_p,
        azimuth_sv=azimuth_sv,
        azimuth_sh=azimuth_sh,
        polar_p=polar_p,
        polar_sv=polar_sv,
        polar_sh=polar_sh,
        azimuth=azimuth,
        polar=polar,
        ray_p=polarization_p,
        ray_sv=polarization_sv,
        ray_sh=polarization_sh,
        ng_p=ng_p,
        ng_sv=ng_sv,
        ng_sh=ng_sh,
    )
    return layer


def thomsen_to_cij(vp_0=2., vs_0=1., epsilon=0., delta=0., gamma=0., rho=1.):
    # % % Calculate  the stiffnesses
    # [Vp0, Vs0, epsilon, delta, gamma]
    # c33 = Ani(1) ^ 2; c11 = c33 * (1 + 2 * Ani(3));
    # c55 = Ani(2) ^ 2; c66 = c55 * (1 + 2 * Ani(5));

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


METHODS = {
    'gpn': thomsen_to_cij_gpn,
    'grechka': thomsen_to_cij,
}


def calculate_layer(
        ani,
        method='gpn',
        axis=1,
        angle=0,
        equidistant_sphere=None,
        a_max=180.,
        a_min=0.,
        p_min=0.,
        p_max=90.,
        n_azi=180,
        n_pol=180,
        tol=1e-15,
):
    rho = np.sqrt(ani.get('rho', 1.))

    _thomsen_to_cij = METHODS.get(method, thomsen_to_cij)

    a = rotation_matrix(angle, axis)
    ani_axis = a.dot(ANI_AXIS)

    c_ij = _thomsen_to_cij(**ani)
    c_ijkl = cij_to_cijkl(c_ij)
    c_ijkl = rotate_cijkl(c_ijkl, a)

    if equidistant_sphere:
        n = init_equidistant_sphere(equidistant_sphere)
    else:
        polar, azimuth = np.meshgrid(
            np.linspace(p_min, p_max, n_pol) * np.pi / 180,
            np.linspace(a_min, a_max, n_azi) * np.pi / 180,
        )
        n = sph2cart(azimuth.ravel(), polar.ravel(), 1, radians=True)

    n = n.T
    g = np.einsum('ijkl, pi, pk -> pjl', c_ijkl, n, n)

    _, s, v = np.linalg.svd(g)

    v = np.real(v)
    s = np.real(np.abs(np.sqrt(s))) / rho

    sign = np.sign(np.einsum('pi, pj, pij -> p', n, n, v) + tol)
    v = np.einsum('pij, p -> pij', v, sign)

    idx_swipe = np.abs(np.einsum('pi, i -> p', v[:, 2], ani_axis)) > 1e-5

    polarization = v.copy()
    v_phase = s.copy()
    polarization[idx_swipe, 1] = v[idx_swipe, 2].copy()
    polarization[idx_swipe, 2] = v[idx_swipe, 1].copy()
    v_phase[idx_swipe, 1] = s[idx_swipe, 2].copy()
    v_phase[idx_swipe, 2] = s[idx_swipe, 1].copy()

    ray = np.einsum('ijkl, pk, pqj, pql, pq -> piq', c_ijkl, n, polarization, polarization, 1 / v_phase)

    polarization_p = polarization[:, 0].copy()
    polarization_sv = polarization[:, 1].copy()
    polarization_sh = polarization[:, 2].copy()

    ray_p = ray[:, :, 0].copy()
    ray_sv = ray[:, :, 1].copy()
    ray_sh = ray[:, :, 2].copy()

    v_phase_p = v_phase[:, 0].copy()
    v_phase_sv = v_phase[:, 1].copy()
    v_phase_sh = v_phase[:, 2].copy()

    azimuth, polar, _ = cart2sph(*n.T, radians=False)
    azimuth_p, polar_p, v_group_p = cart2sph(*ray_p.T, radians=False)
    azimuth_sv, polar_sv, v_group_sv = cart2sph(*ray_sv.T, radians=False)
    azimuth_sh, polar_sh, v_group_sh = cart2sph(*ray_sh.T, radians=False)

    ng_p = sph2cart(azimuth_p.ravel(), polar_p.ravel(), 1, radians=False)
    ng_sv = sph2cart(azimuth_sv.ravel(), polar_sv.ravel(), 1, radians=False)
    ng_sh = sph2cart(azimuth_sh.ravel(), polar_sh.ravel(), 1, radians=False)

    out = dict(
        angle=angle,
        axis=axis,
        ani_axis=ani_axis,
        thomsen=ani,
        c_ij=c_ij,
        c_ij_rot=cijkl_to_cij(c_ijkl),
        polarization_p=polarization_p,
        polarization_sv=polarization_sv,
        polarization_sh=polarization_sh,
        v_group_p=v_group_p,
        v_group_sv=v_group_sv,
        v_group_sh=v_group_sh,
        v_phase_p=v_phase_p,
        v_phase_sv=v_phase_sv,
        v_phase_sh=v_phase_sh,
        azimuth_p=azimuth_p,
        azimuth_sv=azimuth_sv,
        azimuth_sh=azimuth_sh,
        polar_p=polar_p,
        polar_sv=polar_sv,
        polar_sh=polar_sh,
        azimuth=azimuth,
        polar=polar,
        ray_p=ray_p,
        ray_sv=ray_sv,
        ray_sh=ray_sh,
        ng_p=ng_p,
        ng_sv=ng_sv,
        ng_sh=ng_sh,
    )

    return out
