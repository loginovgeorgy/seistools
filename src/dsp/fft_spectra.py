import numpy as np
from copy import deepcopy
from .helpers import cast_input_to_traces


def forward_fourier(x, axis=1, duplicate=True, **kwargs):
    """
    Forward Fourier Transform:
        1 ifftshift
        2 fft
        3 fftshift
    :param x: nD array in time space
    :param axis: axis along to perform fft (axis of time samples)
    :param duplicate: deepcopy object before perform (True/False)
    :return: nD complex array
    """
    if duplicate:
        x = deepcopy(x)
    x = cast_input_to_traces(x)
    x = np.fft.ifftshift(x, axes=axis)
    x = np.fft.fft(x, axis=axis)
    x = np.fft.fftshift(x, axes=axis)
    return x


def backward_fourier(x, axis=1, duplicate=True, only_real=True, **kwargs):
    """
    Inverse Fourier Transform:
        1 ifftshift
        2 ifft
        3 fftshift
    :param x: nD array in frequency space
    :param axis: axis along to perform ifft (axis of time samples)
    :param duplicate: deepcopy object before perform (True/False)
    :param only_real: return only real part of array (True/False)
    :return: nD complex array
    """
    if duplicate:
        x = deepcopy(x)
    # x = cast_input_to_array(x)
    x = np.fft.ifftshift(x, axes=axis)
    x = np.fft.ifft(x, axis=axis)
    x = np.fft.fftshift(x, axes=axis)
    if only_real:
        x = np.real(x)
    return x


def fourier_spectra(x, axis=1, flip=True, duplicate=True, **kwargs):
    """
    Cuts half of fourier spectra. Spectrum has length 'n' - number of samples. The meaning part is only half of one.
    The output array frequency samples are floor(n/2) + 1

    :param x: nD array
    :param axis: axis along to perform cut (axis of time samples)
    :param flip: if forward_fft contains fftshift, it is needed to perform flip (True/False)
    :param duplicate: deepcopy object before perform (True/False)
    :return: return half cutted nD array
    """

    x = forward_fourier(x, axis=axis, duplicate=duplicate)
    ns = x.shape[axis]
    nw = np.int32(np.floor(ns/2) + 1)
    slc = [slice(None)] * len(x.shape)
    slc[axis] = slice(0, nw)
    if flip:
        x = np.flip(x[tuple(slc)], axis=axis)
    return x


def decompose_spectra(s, duplicate=True, **kwargs):
    """
    Split Fourier Spectrum into Amplitude and Phase
    :param s: nD array in frequency space
    :param duplicate: deepcopy object before perform (True/False)
    :return:
        amplitude - abs(s), phase - angle(s)
    """
    if duplicate:
        s = deepcopy(s)
    amplitude = np.abs(s)
    phase = np.angle(s)
    return amplitude, phase


def compose_spectra(amplitude, phase, duplicate=True, **kwargs):
    """
    Compose fill spectrum from Amplitude and Phase parts
    :param amplitude: abs(s)
    :param phase: angle(s)
    :param duplicate: deepcopy object before perform (True/False)
    :return:
        s - amplitude * exp(j*phase)
    """
    if duplicate:
        amplitude = deepcopy(amplitude)
        phase = deepcopy(phase)
    s = amplitude * np.exp(np.array([1.j]) * phase)
    return s


def amplitude_n_phase_spectrum(x, axis=1, unwrap_phase=True, normalize_amplitude=True, **kwargs):
    """
    Calculate Amplitude and Phase spectra of a signal.
    :param x: nD array in time
    :param axis: axis along to perform fft (axis of time samples)
    :param unwrap_phase: Unwrap by changing deltas between values to 2*pi complement. (True/False)
    :param normalize_amplitude: divide Amplitude on num. of time samples (True/False)
    :return:
        amplitude - abs(s), phase - angle(s)
    """
    s = fourier_spectra(x, axis=axis, **kwargs)
    amplitude, phase = decompose_spectra(s, **kwargs)
    if unwrap_phase:
        phase = np.unwrap(phase, axis=axis)
    if normalize_amplitude:
        ns = x.shape[axis]
        amplitude /= ns
    return amplitude, phase


def f_range(dt, ns):
    """
    Create frequency vector
    :param dt: sampling rate
    :param ns: no. of samples
    :return:
    """
    nw = np.int32(np.floor(ns / 2) + 1)
    return np.linspace(0, 1 / 2 / dt, nw)


def fft_interpolation(x, dt1, dt2, axis=1, duplicate=True):
    """
    Provides interpolation from sampling dt1 to sampling dt2
    :param x:
    :param dt1:
    :param dt2:
    :param axis:
    :param duplicate:
    :return:
    """
    if duplicate:
        x = deepcopy(x)

    ns1 = x.shape[axis]
    x_shape = list(x.shape)

    if np.mod(ns1, 2):
        x_shape[axis] = 1
        add_zeros = np.zeros(x_shape)
        x = np.concatenate([x, add_zeros], axis=axis)

    nw1 = np.int32(np.floor(ns1 / 2) + 1)
    ns2 = np.int32(np.floor(dt1 * ns1 / dt2))
    nw2 = np.int32(np.floor(ns2 / 2) + 1)

    dn = np.int32(nw2 - nw1)

    x_shape[axis] = dn
    add_zeros = np.zeros(x_shape)

    s1 = forward_fourier(x) / ns1
    s2 = np.concatenate([add_zeros, s1, add_zeros], axis=axis)
    x2 = backward_fourier(s2 * ns2)

    return x2


def create_band_pass_filter(ns, dt, bound):
    """
    Create band pass filter
    :param ns: num. of time samples
    :param dt: sampling rate
    :param bound: filter bounds in Hz
    :return: f - filter, 1D array
    """
    b = np.array(bound)
    if len(b) != 4:
        raise Exception(
            'Filter bounds must have 4 elements.'
            'The shape of bounds was {}.'.format(b.shape)
        )
    if np.any(np.diff(b) < 0):
        raise Exception(
            'Filter bounds must increase.'
            'The bounds were {}.'.format(b)
        )
    nw = np.int32(np.floor(ns/2)) + 1
    b = np.int32(np.round(b*(ns-1)*dt))
    b[b < 0] = 0
    b[b > (nw-1)] = nw
    f = np.zeros(nw)
    f[b[0]: b[1]] = np.linspace(0, 1, b[1]-b[0])
    f[b[1]: b[2]] = 1
    f[b[2]: b[3]] = np.linspace(1, 0, b[3]-b[2])
    return f


def create_band_reject_filter(ns, dt, bound):
    """
    Create band pass filter
    :param ns: num. of time samples
    :param dt: sampling rate
    :param bound: filter bounds in Hz
    :return: f - filter, 1D array
    """
    b = np.array(bound)
    if len(b) != 4:
        raise Exception(
            'Filter bounds must have 4 elements.'
            'The shape of bounds was {}.'.format(b.shape)
        )
    if np.any(np.diff(b) < 0):
        raise Exception(
            'Filter bounds must increase.'
            'The bounds were {}.'.format(b)
        )
    nw = np.int32(np.floor(ns/2)) + 1
    b = np.int32(np.round(b*(ns-1)*dt))
    b[b < 0] = 0
    b[b > (nw-1)] = nw
    f = np.ones(nw)
    f[b[0]: b[1]] = np.linspace(1, 0, b[1]-b[0])
    f[b[1]: b[2]] = 0
    f[b[2]: b[3]] = np.linspace(0, 1, b[3]-b[2])
    return f


def apply_filter(x, f, axis=1):
    """
    Apply 1D filter along time axis
    :param x: input signal, nD Array
    :param f: 1D array filter
    :param axis: axis along to perform filtering (axis of time samples)
    :return: x - filtered signal
    """

    x = cast_input_to_traces(x)
    f = cast_input_to_traces(f, ndmin=1)

    ns = x.shape[axis]
    nw = np.int32(np.floor(ns/2) + 1)

    j = np.abs(np.arange(-nw + 1, nw)[:ns])

    f = f.ravel()[j]
    f = cast_input_to_traces(f, ndmin=len(x.shape))
    f = f.transpose(
        np.roll(
            np.arange(len(x.shape)),
            axis+1
        )
    )

    nf = len(f.ravel())

    if nf != ns:
        raise Exception(
            'Filter length must be same as signal.'
            'The shape of filter was {}.'
            'The shape of signal was {}. Signal samples was {}'.format(f.shape, x.shape, x.shape[axis])
        )

    return backward_fourier(forward_fourier(x, axis=axis)*f, axis=axis)


def apply_band_pass(x, dt, bound, axis=1):
    """
    Create and apply band_pass filter

    :param x: input signal, nD Array
    :param dt: sampling rate
    :param bound: filter bounds in Hz
    :param axis: axis along to perform filtering (axis of time samples)
    :return: x - filtered signal
    """
    x = cast_input_to_traces(x)
    ns = x.shape[axis]
    f = create_band_pass_filter(ns, dt, bound)
    return apply_filter(x, f, axis=axis)


def apply_band_reject(x, dt, bound, axis=1):
    """
    Create and apply band_pass filter

    :param x: input signal, nD Array
    :param dt: sampling rate
    :param bound: filter bounds in Hz
    :param axis: axis along to perform filtering (axis of time samples)
    :return: x - filtered signal
    """
    x = cast_input_to_traces(x)
    ns = x.shape[axis]
    f = create_band_reject_filter(ns, dt, bound)
    return apply_filter(x, f, axis=axis)
