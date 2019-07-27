import numpy as np
from copy import deepcopy
from .helpers import cast_input_to_array

EPS = 1e-16


def calculate_rms_amplitude(traces, window=301, axis=1, squeeze_window=False, duplicate=True):
    """
    Calculate RMS of a signal in 'central' window of time sample
    :param traces: input seismic traces, nD array
    :param window: rms window length, must be odd
    :param axis: axis along to calculate rms (axis of time samples)
    :param squeeze_window: skip time interval that lower then window
    :param duplicate: deepcopy object before perform (True/False)
    :return:
        g - root mean square of signal
    """
    if not window % 2:
        raise Exception(
            'window length must be odd.'
            'The window was {}.'.format(window)
        )
    if duplicate:
        traces = deepcopy(traces)

    traces = cast_input_to_array(traces)

    g = np.zeros(traces.shape)
    ns = traces.shape[axis]

    slc_g = [slice(None)] * len(traces.shape)
    slc_t = [slice(None)] * len(traces.shape)

    window_right = np.int32(np.floor(window / 2))
    window_left = window_right + 1

    if squeeze_window:
        range_start = window
        range_end = ns - window
    else:
        range_start = 0
        range_end = ns

    for i in range(range_start, range_end):
        jfr = max([(i - window_left + 1), 0])
        jto = min([ns, (i + window_right)])

        slc_t[axis] = slice(jfr, jto)
        slc_g[axis] = slice(i, i + 1)

        temp = traces[tuple(slc_t)]
        measure = temp.std(axis=axis, keepdims=True)
        g[tuple(slc_g)] = measure

    return g


def calculate_gain_correction(g, des_std=1., duplicate=True):
    """
    Calculate Gain correction to desired STD
    :param g: characteristic (For example RMS), nD array
    :param des_std: desired std of a signal, value, float
    :param duplicate: deepcopy object before perform (True/False)
    :return:
        g - gain correction
    """
    if duplicate:
        g = deepcopy(g)
    des_std = np.float32(des_std)
    correction = des_std / (g + EPS)
    return correction


def apply_gain_correction(traces, correction, duplicate=True, cast=True):
    """
    Apply gain corection to seismic traces.
    :param traces: input seismic traces, nD array
    :param correction: correction, nD array,
    :param duplicate: deepcopy object before perform (True/False)
    :param cast: cast to expected dtype (True/False)
    :return: corrected traces
    """
    if duplicate:
        traces = deepcopy(traces)
        correction = deepcopy(correction)

    if cast:
        traces = cast_input_to_array(traces)
        correction = cast_input_to_array(correction)

    return traces*correction


def apply_rms_correction(traces, window=301, des_std=1., axis=1, squeeze_window=False):
    """
    Calculate and apply RMS gain correction
    :param traces: input seismic traces, nD array
    :param window: rms window length, must be odd
    :param des_std: desired std of a signal, value, float
    :param axis: axis along to calculate rms (axis of time samples)
    :param squeeze_window: skip time interval that lower then window
    :return:
        corrected by RMS traces
    """
    traces = deepcopy(traces)
    g = calculate_rms_amplitude(traces, window=window, axis=axis, squeeze_window=squeeze_window, duplicate=False)
    correction = calculate_gain_correction(g, des_std=des_std, duplicate=False)
    return apply_gain_correction(traces, correction, duplicate=False, cast=False)