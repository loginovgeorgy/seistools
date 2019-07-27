import numpy as np
from .helpers import moving_average_1d, cast_input_to_array
from .normalizing import normalize_data
from copy import deepcopy

EPS = 1e-16


def detection_stalta(x, window_short, window_long, axis=1, drop_edges=True):
    """
    Calculate STA/LTA detection function
    :param x: signal
    :param window_short:
    :param window_long:
    :param axis: axis along to perform filtering (axis of time samples)
    :param drop_edges: drop region with partial window overlapping
    :return:
        detection function
    """
    x = np.squeeze(x)
    short = moving_average_1d(x ** 2, window_short, window_type='left', axis=axis)
    long = moving_average_1d(x ** 2, window_long, window_type='left', axis=axis)
    det = short / (long + EPS)
    det = np.abs(np.gradient(det))

    if drop_edges:
        slc = [slice(None)] * len(x.shape)
        slc[axis] = slice(0, window_long)
        det[tuple(slc)] = 0

    return det


def detection_mer(x, window, axis=1, drop_edges=True):
    """
    Calculate MER detection function
    :param x: signal
    :param window:
    :param axis: axis along to perform filtering (axis of time samples)
    :param drop_edges: drop region with partial window overlapping
    :return:
        detection function
    """
    x = deepcopy(x)
    x = cast_input_to_array(x)

    left = moving_average_1d(x ** 2, window, axis=axis, window_type='left')
    right = moving_average_1d(x ** 2, window, axis=axis, window_type='right')

    det = right / (left + EPS)
    det = (np.abs(x) * det) ** 3

    if drop_edges:
        ns = x.shape[axis]
        slc = [slice(None)] * len(x.shape)
        slc[axis] = slice(0, window)
        det[tuple(slc)] = 0
        slc[axis] = slice(ns-window, ns)
        det[tuple(slc)] = 0

    return det


def detection_em(x, window, axis=1, drop_edges=True):
    """
    Calculate EM detection function
    :param x: signal
    :param window:
    :param axis: axis along to perform filtering (axis of time samples)
    :param drop_edges: drop region with partial window overlapping
    :return:
        detection function
    """
    x = deepcopy(x)
    x = cast_input_to_array(x)

    s_diff = np.gradient(x, axis=axis)
    h = moving_average_1d(np.abs(s_diff), window, axis=axis, window_type='left')
    h_log = np.log10(h + EPS)
    det = np.gradient(h_log, axis=axis)
    det = np.abs(det)

    if drop_edges:
        slc = [slice(None)] * len(x.shape)
        slc[axis] = slice(0, window)
        det[tuple(slc)] = 0

    return det


def detection_threshold(x, threshold, axis=1, normalize=False, replace_threshold=False, error_picks=-1):
    """
    Detect argument of value per threshold
    :param x:
    :param threshold:
    :param axis:
    :param normalize:
    :param replace_threshold: provide picking anyway
    :param error_picks: cast errors to value
    :return:
    """
    x = deepcopy(x)
    det = cast_input_to_array(x)

    if normalize:
        det = normalize_data(det, axis=axis, shift_type=None, scale_type='max')

    _det = deepcopy(det)
    _det -= threshold
    picks = _det.argmax(axis=axis, keepdims=True)

    if not replace_threshold:
        value = det.max(axis=axis, keepdims=True)
        idx = value < threshold
        picks[idx] = error_picks

    picks = np.squeeze(picks)
    picks = np.array(picks, ndmin=1)
    return picks


def by_threshold(x, threshold, axis=1, weight=1, shift=1):
    x = x.copy()
    x -= threshold
    x = np.sign(x)
    x *= weight
    idx = x.argmax(axis=axis)
    idx += shift
    return idx
