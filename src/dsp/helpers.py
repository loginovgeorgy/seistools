import numpy as np
from copy import deepcopy
# from scipy.signal import convolve
from scipy.signal import fftconvolve as convolve


def cast_to_3c_traces(traces, nc=3):
    nr0 = traces.shape[0]
    ns = traces.shape[1]

    if nr0 % nc:
        raise ValueError('The given trace shape {} cant be casted to (-1, {}, {})'.format(traces.shape, ns, nc))

    nr = int(nr0 / nc)

    traces = traces.reshape(nr, nc, -1)
    return traces.transpose((0, 2, 1))


def cast_input_to_traces(x, ndmin=2):
    """
    It is better to check input data type and shape. To prevent problems, array must be at least 2D,
    where time axis equals 1.
    :param x: list or array
    :param ndmin: minimal array dimension
    :return: np.array(x, ndmin=ndmin)
    """
    # x = np.array(x)
    # x = np.squeeze(x)
    x = deepcopy(x)
    x = np.array(x, ndmin=ndmin, dtype=np.float32)
    return x


def add_zeros(data, nr, ns):
    data = cast_input_to_traces(data)

    if data.shape[0] < nr:
        temp = np.zeros((nr - data.shape[0], data.shape[1]))
        data = np.vstack((data, temp))
    if data.shape[1] < ns:
        temp = np.zeros((data.shape[0], ns - data.shape[1]))
        data = np.hstack((data, temp))
    if data.shape[0] > nr:
        data = data[:nr]
    if data.shape[1] > ns:
        data = data[:ns]
    return data


def matrix_delta(tau, ns=None):
    """
    Create matrix of delta functions at samples tau
    :param tau: array of int
    :param ns: maximum number of samples, must be higher then 0
    :return:
        array of shape len(tau) * ns
    """
    if isinstance(tau, (float, bool, int)):
        tau = [tau]

    tau = np.array(tau)
    tau = np.int32(tau)
    if ns:
        ns = np.int32(ns)
        if ns < 1:
            raise Exception(
                'ns must be more then 0.'
                'The given ns was {}'.format(ns)
            )

        tau[tau >= ns] = ns - 1
    else:
        ns = tau.max()

    nd_min = len(tau.shape)

    matrix = np.zeros(list(tau.shape) + [ns])
    slc_w = [slice(None)] * (nd_min + 1)

    for i in range(nd_min):
        tmp = np.arange(tau.shape[i])
        tmp = np.array(tmp, ndmin=nd_min)
        tmp = np.transpose(tmp, np.roll(np.arange(nd_min), -(i + nd_min - 1)))

        slc_w[i] = tmp

    slc_w[nd_min] = tau

    matrix[tuple(slc_w)] = 1
    return matrix


def matrix_heaviside(tau, ns=None):
    """
    Create matrix of heaviside functions at samples tau
    :param tau: array of int
    :param ns: maximum number of samples, must be higher then 0
    :return:
        array of shape len(tau) * ns
    """
    matrix = matrix_delta(tau, ns)
    return np.cumsum(matrix, axis=1)


def moving_average_1d(x, window, axis=1, window_type='left'):
    """
    Calculate Moving Average of signal along certain axis.
    :param x: input array
    :param window: window length
    :param axis:
    :param window_type:
        'left'
        'right'
        'center'
    :return:
    """

    window = np.int32(window)

    if window_type not in ['left', 'right', 'center']:
        raise Exception(
            'Must choose one of three window types: "left", "right", "center"'
            'The given window type is {}'.format(window_type)
        )

    if (not window % 2) & (window_type == 'center'):
        raise Exception(
            'window length must be odd.'
            'The window was {}.'.format(window)
        )

    x = cast_input_to_traces(x, ndmin=2)
    ns = x.shape[axis]

    f = np.ones(window)/window
    f = cast_input_to_traces(f, ndmin=len(x.shape))
    f = f.transpose(
        np.roll(
            np.arange(len(x.shape)),
            axis + 1
        )
    )

    if window_type == 'center':
        mode = 'same'
    else:
        mode = 'full'

    if window_type == 'left':
        x = np.flip(x, axis=axis)

    average = convolve(x, f, mode=mode, axes=axis)
    average = np.real(average)

    slc = [slice(None)] * len(x.shape)

    if window_type == 'left':
        average = np.flip(average, axis=axis)

    if window_type == 'right':
        slc[axis] = slice(window - 1, ns + window)
    else:
        slc[axis] = slice(0, ns)

    return average[tuple(slc)]


def calculate_convolution(x, f, axis=1, mode='full'):
    x = cast_input_to_traces(x, ndmin=2)
    ns = x.shape[axis]

    f = cast_input_to_traces(f, ndmin=len(x.shape))
    f = f.transpose(
        np.roll(
            np.arange(len(x.shape)),
            axis + 1
        )
    )

    return convolve(x, f, mode=mode, axes=axis)


def edge_preserve_smoothing(signal, window, verbose=False):
    # TODO Rebuild to boost speed
    if window <= 0:

        return signal
    ns = len(signal)
    eps_avg = np.zeros(ns)

    for i in range(ns):
        jfrom = max([i - window + 1, 0])
        interval = np.arange(jfrom, i+1)
        _avg = np.zeros(len(interval))
        _std = np.zeros(len(interval))
        if verbose:
            print(i, interval)
        for j, jfrom in enumerate(interval):
            jto = min([jfrom + window, ns])

            temp_signal = signal[jfrom:jto]
            _avg[j] = temp_signal.mean()
            _std[j] = temp_signal.std()
            if verbose:
                print("\t {} {:.2f} {:.2f}".format(str((np.arange(jfrom,jto))), _avg[j], _std[j]))

#         eps_avg[i] = _avg[np.where(_std==_std.min())[0]].min()
        eps_avg[i] = _avg[_std.argmin()]
    return eps_avg


def polarization_analysis(hodogram):
    _, eig_val, eig_vec = np.linalg.svd(hodogram)
    eig_val = eig_val ** 2
    eig_val = eig_val / np.nanmax(eig_val)

    eig_idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[eig_idx]
    eig_vec = eig_vec[:, eig_idx]

    vector3 = np.cross(eig_vec[:, 0], eig_vec[:, 1])
    dot_prod = np.dot(vector3, eig_vec[:, 2])

    if dot_prod > 0:
        eig_vec[:, 2] *= -1

    polarization = eig_vec
    this_rot = polarization

    polar_angle = np.abs(np.arctan2(np.linalg.norm(this_rot[:1, 0]), this_rot[2, 0]))
    # % PolarAng = acos(this_rot(3, jw) / norm(this_rot(:, jw), 2));
    azimuth = np.arctan2(this_rot[1, 0], this_rot[0, 0])

    azm_vec = np.array([-this_rot[1, 0], this_rot[0, 0], 0])
    pol_vec = np.cross(azm_vec, this_rot[:, 0])
    nrm_val = np.pi * eig_val / np.max(eig_val)

    polar_angle_std = np.abs(np.dot(pol_vec, nrm_val[1] * this_rot[:, 1])) + \
                      np.abs(np.dot(pol_vec, nrm_val[2] * this_rot[:, 2]))

    azimuth_std = np.abs(np.dot(azm_vec, nrm_val[1] * this_rot[:, 1])) + \
                  np.abs(np.dot(azm_vec, nrm_val[2] * this_rot[:, 2]))
    linearity = ((eig_val[0] - eig_val[1]) ** 2 + (eig_val[0] - eig_val[2]) ** 2 +
                 (eig_val[1] - eig_val[2]) ** 2) / (2 * sum(eig_val) ** 2)

    return azimuth, azimuth_std, polar_angle, polar_angle_std, linearity

