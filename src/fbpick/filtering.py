import numpy as np
from .fft_spectra import forward_fourier, backward_fourier
from .helpers import cast_input_to_array


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


def apply_filter(x, f, axis=1):
    """
    Apply 1D filter along time axis
    :param x: input signal, nD Array
    :param f: 1D array filter
    :param axis: axis along to perform filtering (axis of time samples)
    :return: x - filtered signal
    """

    x = cast_input_to_array(x)
    f = cast_input_to_array(f, ndmin=1)

    ns = x.shape[axis]
    nw = np.int32(np.floor(ns/2) + 1)

    j = np.abs(np.arange(-nw + 1, nw)[:ns])

    f = f.ravel()[j]
    f = cast_input_to_array(f, ndmin=len(x.shape))
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
    :param x: input signal, nD Array
    :param dt: sampling rate
    :param bound: filter bounds in Hz
    :param axis: axis along to perform filtering (axis of time samples)
    :return: x - filtered signal
    """
    x = cast_input_to_array(x)
    ns = x.shape[axis]
    f = create_band_pass_filter(ns, dt, bound)
    return apply_filter(x, f, axis=axis)
