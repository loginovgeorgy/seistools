import numpy as np
from copy import deepcopy
from itertools import product
from .utils import *

SYMMETRY_AXIS = np.array([0, 0, 1])


def _cast_input_vector(n, vector_type="wave normal"):
    n = np.array(n)
    n = np.squeeze(n)
    n = np.array(n, dtype=np.float32, ndmin=2)
    if n.shape[1] != 3:
        n = n.T

    if n.shape[1] != 3:
        raise ValueError(
            "Input {} must be 1D array of 3 elements or 2D array like (N, 3)."
            "The given was {}".format(vector_type, n.shape)
        )

    return n


def calculate_travel_time(
        sou,
        rec,
        thomsen_params,
        rotation_axis=1,
        rotation_angle=0,
        symmetry_axis=None,
        tol=1e-15,
        verbose=True,
):
    """
    Calculate travel times for homogeneous anisotropic media for
    given sources and receivers locations. If number of sources is more than one,
    and not equal to number of receivers, then the output will be given for each pair of source-receiver.

    :param sou: source location (s) - 1D array of 3 elements or 2D (N, 3) array
    :param rec: receiver location (s) - 1D array of 3 elements or 2D (M, 3) array
    :param thomsen_params: [Vp0, Vs0, epsilon, delta, gamma, rho (density)]
    :param rotation_axis: axis to perform rotation of cij (x - 0, y- 1, z - 2), default - 1
    :param rotation_angle: angle to perform rotation of cij (degrees, from 0 to 360), default - 0
    :param symmetry_axis: 1D array of 3 elements, necessary to distinguish the behavior of SV and SH polarizations,
    default - vertical ([0,0,1])
    :param tol: tolerance (1e-15)
    :param verbose: print parameters of calculations
    :return: dict for {'P', 'SV' 'SH'} with travel times with shape (N, M), where N - number of sources
    and M - number of receivers
    """

    sou = _cast_input_vector(sou, vector_type='source location')
    rec = _cast_input_vector(rec, vector_type='receiver location')

    n_sou = sou.shape[0]
    n_rec = rec.shape[0]

    if verbose:
        msg = (
            "No. of sources {} \n"
            "No. of receivers {} \n"
            "Thomsen parameters: {}"
        )
        print(msg)

    n = rec[None, ...] - sou[:, None, :]
    n = n.reshape(-1, 3)
    distance = np.sqrt((n ** 2).sum(axis=1, keepdims=True))

    attributes = calculate_attributes(
        n,
        thomsen_params,
        rotation_angle=rotation_angle,
        rotation_axis=rotation_axis,
        symmetry_axis=symmetry_axis,
        tol=tol
    )

    time = {wave: distance / attributes[wave]["v_group"] for wave in ['P', 'SV', 'SH']}
    time = {wave: time[wave].reshape(n_sou, n_rec) for wave in ['P', 'SV', 'SH']}

    return time


def calculate_layer_attributes(
        thomsen_params,
        symmetry_axis=None,
        rotation_axis=1,
        rotation_angle=0,
        equidistant_sphere=None,
        a_max=180.,
        a_min=0.,
        p_min=0.,
        p_max=90.,
        n_azi=180,
        n_pol=180,
        tol=1e-15,
):
    """
    Calculate attributes of wave propagation along the given space of wave directions
    for homogeneous anisotropic media.
    :param thomsen_params: dictionary or list of [Vp0, Vs0, epsilon, delta, gamma, rho (density)]
    :param symmetry_axis: 1D array of 3 elements, necessary to distinguish the behavior of SV and SH polarizations
    default - vertical ([0,0,1])
    :param rotation_angle: angle to perform rotation of cij (degrees, from 0 to 360), default - 0
    :param rotation_axis: axis to perform rotation of cij (x - 0, y- 1, z - 2), default - 1
    :param equidistant_sphere: None or int, to initialize vectors on an equidistant sphere
    if None, then operate to following properties:
    :param a_max: maximum azimuth value (degrees)
    :param a_min: minimum azimuth value (degrees)
    :param p_min: minimum polar angle value (degrees)
    :param p_max: maximum polar angle value (degrees)
    :param n_azi: number of azimuth samples (int)
    :param n_pol: number of polar angle samples (int)
    :param tol: tolerance (1e-15)
    :return: dict for
    {'P', 'SV' 'SH'} waves with corresponding dict with keys, where N - number of given wave normals
        - polarization: shape (N, 3),
        - v_group: (ray velocity), shape (N, 1),
        - v_phase: (phase velocity), shape (N, 1),
        - azimuth: (ray/polarization azimuth), shape (N, 1),
        - polar: (ray/polarization polar angle), shape (N, 1),
        - ray: (ray vector), shape (N, 3),
        - n: (phase vector), shape (N, 3),

    {'cij'} input properties of media
        - rotation_angle: float,
        - rotation_axis: int,
        - symmetry_axis: 1D array,
        - thomsen_params: dict,
        - c_ij_original: stiffness tensor defined by thomsen_params
        - c_ij_rotated: stiffness tensor cij rotated to given properties

    {'n') input wave normal, to check how module see directivity of given normals
        - n: given normals, shape (N, 3),
        - azimuth: shape (N, 1),
        - polar: shape (N, 3),
    """

    if equidistant_sphere:
        n = init_equidistant_sphere(equidistant_sphere)
    else:
        polar, azimuth = np.meshgrid(
            np.linspace(p_min, p_max, n_pol) * np.pi / 180,
            np.linspace(a_min, a_max, n_azi) * np.pi / 180,
            )
        n = sph2cart(azimuth.ravel(), polar.ravel(), 1, radians=True)

    return calculate_attributes(
        n,
        thomsen_params,
        rotation_angle=rotation_angle,
        rotation_axis=rotation_axis,
        symmetry_axis=symmetry_axis,
        tol=tol
    )


def calculate_attributes(n, thomsen_params, rotation_angle=0, rotation_axis=1, symmetry_axis=None, tol=1e-15):
    """
    Calculate attributes of wave propagation for homogeneous anisotropic media for
    given wave normal 'n'.
    :param n: wave normal, 1D array of 3 elements or 2D (N, 3) array
    :param thomsen_params: dictionary or list of [Vp0, Vs0, epsilon, delta, gamma, rho (density)]
    :param rotation_angle: angle to perform rotation of cij (degrees, from 0 to 360), default - 0
    :param rotation_axis: axis to perform rotation of cij (x - 0, y- 1, z - 2), default - 1
    :param symmetry_axis: 1D array of 3 elements, necessary to distinguish the behavior of SV and SH polarizations
    default - vertical ([0,0,1])
    :param tol: tolerance (1e-15)
    :return: dict for
    {'P', 'SV' 'SH'} waves with corresponding dict with keys, where N - number of given wave normals
        - polarization: shape (N, 3),
        - v_group: (ray velocity), shape (N, 1),
        - v_phase: (phase velocity), shape (N, 1),
        - azimuth: (ray/polarization azimuth), shape (N, 1),
        - polar: (ray/polarization polar angle), shape (N, 1),
        - ray: (ray vector), shape (N, 3),
        - n: (phase vector), shape (N, 3),

    {'cij'} input properties of media
        - rotation_angle: float,
        - rotation_axis: int,
        - symmetry_axis: 1D array,
        - thomsen_params: dict,
        - c_ij_original: stiffness tensor defined by thomsen_params
        - c_ij_rotated: stiffness tensor cij rotated to given properties

    {'n') input wave normal, to check how module see directivity of given normals
        - n: given normals, shape (N, 3),
        - azimuth: shape (N, 1),
        - polar: shape (N, 3),

    """

    n = _cast_input_vector(n)
    distance = np.sqrt((n ** 2).sum(axis=1, keepdims=True))
    n /= (distance + tol)

    if isinstance(symmetry_axis, type(None)):
        symmetry_axis = SYMMETRY_AXIS

    rho = np.sqrt(thomsen_params.get('rho', 1.))

    a = rotation_matrix(rotation_angle, rotation_axis)
    symmetry_axis = a.dot(symmetry_axis)

    c_ij = thomsen_to_cij(**thomsen_params)
    c_ijkl = cij_to_cijkl(c_ij)
    c_ijkl = rotate_cijkl(c_ijkl, a)

    g = np.einsum('ijkl, pi, pk -> pjl', c_ijkl, n, n)

    # svd performs automatic sorting by eigenvalues and does not strongly effects on calculation speed
    _, s, v = np.linalg.svd(g)

    # prevent not expected errors
    v = np.real(v)
    s = np.sqrt(s)
    s = np.abs(s)
    s = np.real(s)
    s /= rho

    sign = np.sign(np.einsum('pi, pj, pij -> p', n, n, v) + tol)
    v = np.einsum('pij, p -> pij', v, sign)

    # check which wave corresponds to SV and SH
    idx_swipe = np.abs(np.einsum('pi, i -> p', v[:, 2], symmetry_axis)) > 1e-5

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

    azimuth_p, polar_p, v_group_p = cart2sph(*ray_p.T, radians=False)
    azimuth_sv, polar_sv, v_group_sv = cart2sph(*ray_sv.T, radians=False)
    azimuth_sh, polar_sh, v_group_sh = cart2sph(*ray_sh.T, radians=False)

    ng_p = sph2cart(azimuth_p.ravel(), polar_p.ravel(), 1, radians=False)
    ng_sv = sph2cart(azimuth_sv.ravel(), polar_sv.ravel(), 1, radians=False)
    ng_sh = sph2cart(azimuth_sh.ravel(), polar_sh.ravel(), 1, radians=False)

    azimuth_n, polar_n, _ = cart2sph(*n.T, radians=False)

    out = {
        "P": dict(
            polarization=polarization_p,
            v_group=v_group_p,
            v_phase=v_phase_p,
            azimuth=azimuth_p,
            polar=polar_p,
            ray=ray_p,
            n=ng_p,
        ),
        "SV": dict(
            polarization=polarization_sv,
            v_group=v_group_sv,
            v_phase=v_phase_sv,
            azimuth=azimuth_sv,
            polar=polar_sv,
            ray=ray_sv,
            n=ng_sv,
        ),
        "SH": dict(
            polarization=polarization_sh,
            v_group=v_group_sh,
            v_phase=v_phase_sh,
            azimuth=azimuth_sh,
            polar=polar_sh,
            ray=ray_sh,
            n=ng_sh,
        ),
        'n': dict(
            n=n,
            azimuth=azimuth_n,
            polar=polar_n,
        ),
        'cij': dict(
            rotation_angle=rotation_angle,
            rotation_axis=rotation_axis,
            symmetry_axis=symmetry_axis,
            thomsen_params=thomsen_params,
            c_ij_original=c_ij,
            c_ij_rotated=cijkl_to_cij(c_ijkl),
        )
    }

    return out
