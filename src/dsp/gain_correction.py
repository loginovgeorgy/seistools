import numpy as np
from copy import deepcopy
from collections import defaultdict
from .helpers import cast_input_to_traces, moving_average_1d

EPS = 1e-26


def _mean(x, axis):
    return np.nanmean(x, axis=axis, keepdims=True)


def _median(x, axis):
    return np.nanmedian(x, axis=axis, keepdims=True)


def _max(x, axis):
    return np.nanmax(x, axis=axis, keepdims=True)


def _min(x, axis):
    return np.nanmax(x, axis=axis, keepdims=True)


def _std(x, axis):
    return np.nanstd(x, axis=axis, keepdims=True)


def _maxabs(x, axis):
    return np.nanmax(np.abs(x), axis=axis, keepdims=True)


def _minmax(x, axis):
    return np.nanmax(x, axis=axis, keepdims=True) - np.nanmin(x, axis=axis, keepdims=True)


def _minmaxabs(x, axis):
    return np.nanmax(np.abs(x), axis=axis, keepdims=True) - np.nanmin(np.abs(x), axis=axis, keepdims=True)


def _zero_shift(*args):
    return 0


def _ones_scale(*args):
    return 1


TOOLS = {
    'mean': _mean,
    'median': _median,
    'max': _max,
    'min': _min,
    'std': _std,
    'maxabs': _maxabs,
    'minmax': _minmax,
    'minmaxabs': _minmaxabs,
    'zero': _zero_shift,
    'one': _ones_scale,
    }

SCALING = defaultdict(lambda: _ones_scale)
SCALING.update(TOOLS)
SHIFTING = defaultdict(lambda: _zero_shift)
SHIFTING .update(TOOLS)


def calculate_scale(x, axis=1, scale_type='max'):
    _func = SCALING[scale_type]
    return _func(x, axis)


def calculate_shift(x, axis=1, shift_type='mean'):
    _func = SHIFTING[shift_type]
    return _func(x, axis)


def normalize_traces_by_std(traces, window, window_type='center', axis=1, eps=EPS):
    """
    Normalize traces in a running window
    :param traces: input seismic
    :param window: window length
    :param window_type: 'center', 'left', 'right'
    :param axis: axis along to perform normalization
    :param eps: epsilon
    :return:
    """

    average = moving_average_1d(traces, window, window_type=window_type, axis=axis)
    std = moving_average_1d((traces - average) ** 2, window, window_type=window_type, axis=axis)
    return traces / (np.sqrt(np.abs(std)) + eps)


def normalize_traces(
        x,
        axis=1,
        shift_type='mean',
        scale_type='maxabs',
        duplicate=True,
        cast=True,
        calc_scale_after_shift=True,
        eps=EPS,
):
    """
    Perform data normalization along set axis, to bring values in certain interval.
    :param x: input data, nD array
    :param axis: axis along to perform scaling (axis of time samples)
    :param shift_type: if None - 0
    :param scale_type: if None - 1
        None - cast to default
        'mean': np.mean(x, keepdims=True),
        'median': _median,
        'max': _max,
        'min': _min,
        'std': _std,
        'maxabs': _maxabs,
        'minmax': _minmax,
        'minmaxabs': _minmaxabs,
        'zero': _zero_shift,
        'one': _ones_scale,
    :param duplicate: deepcopy object before perform (True/False)
    :param cast: cast to expected dtype (True/False)
    :param calc_scale_after_shift:  bool, calculate scale after apply shift
    :param eps: epsilon
    :return:
    """

    if duplicate:
        x = deepcopy(x)

    if cast:
        x = cast_input_to_traces(x)

    if not calc_scale_after_shift:
        shift = calculate_shift(x, axis=axis, shift_type=shift_type)
        scale = calculate_scale(x, axis=axis, scale_type=scale_type)

        x = (x - shift) / (scale + eps)
    else:
        x -= calculate_shift(x, axis=axis, shift_type=shift_type)
        x /= (calculate_scale(x, axis=axis, scale_type=scale_type) + eps)

    return x


def calculate_rms_amplitude(
        traces,
        window=301,
        axis=1,
        squeeze_window=False,
        duplicate=True
):
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

    traces = cast_input_to_traces(traces)

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


def calculate_gain_correction(
        g,
        des_std=1.,
        duplicate=True
):
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
        traces = cast_input_to_traces(traces)
        correction = cast_input_to_traces(correction)

    return traces*correction


def apply_rms_correction(
        traces,
        window=301,
        des_std=1.,
        axis=1,
        squeeze_window=False
):
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

