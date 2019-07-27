import numpy as np
from .helpers import moving_average_1d, cast_input_to_traces
from .gain_correction import normalize_traces
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

    det = np.abs(np.gradient(det, axis=axis))

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
    x = cast_input_to_traces(x)

    left = moving_average_1d(x ** 2, window, axis=axis, window_type='left')
    right = moving_average_1d(x ** 2, window, axis=axis, window_type='right')

    det = right / (left + EPS)
    det = (np.abs(x) * det) ** 3

    if drop_edges:
        ns = x.shape[axis]
        slc = [slice(None)] * len(x.shape)
        slc[axis] = slice(0, window)
        det[tuple(slc)] = 0
        slc[axis] = slice(ns - window, ns)
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
    x = cast_input_to_traces(x)

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


def detect_by_threshold(x, threshold, axis=1, normalize=False, replace_threshold=False, error_picks=-1):
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
    det = cast_input_to_traces(x)

    if normalize:
        det = normalize_traces(det, axis=axis, shift_type=None, scale_type='max')

    _det = deepcopy(det)
    _det -= threshold
    picks = _det.argmax(axis=axis)

    if not replace_threshold:
        value = det.max(axis=axis, keepdims=True)
        idx = value < threshold
        picks[idx] = error_picks

    picks = np.squeeze(picks)
    picks = np.array(picks, ndmin=1)
    return picks


def trigger_detection_function(traces, window=200, smooth=401, threshold=2, eps=EPS):

    traces = cast_input_to_traces(traces)

    left = moving_average_1d(traces ** 2, window, axis=1, window_type='left')
    right = moving_average_1d(traces ** 2, window, axis=1, window_type='right')
    s = right / (left + eps)
    s[:, :window] = 1
    s[:, -window::1] = 1

    s0 = moving_average_1d(s, smooth, axis=1, window_type='left')
    s0 = s0.mean(axis=2) > threshold
    s0 = s0.sum(axis=0)
    return s0


def get_detection_spikes(x):

    d = np.diff(x.copy(), append=0)

    if np.abs(d).max() == 0:
        return None, None, None

    idx_plus = np.where(d > 0)[0]
    idx_minus = np.where(d < 0)[0]

    idx_right = idx_minus[..., None] - idx_plus[None, ...]
    idx_right[idx_right <= 0] = len(x)
    idx_right = idx_right.argmin(axis=0)
    idx_right = np.sort(np.unique(idx_right))

    idx_left = idx_minus[idx_right, None] - idx_plus[None, ...]
    idx_left[idx_left <= 0] = len(x)
    idx_left = idx_left.argmin(axis=1)

    pick = np.stack([idx_plus[idx_left], idx_minus[idx_right]], axis=-1)

    pick = pick.mean(axis=1)
    pick = np.round(pick)
    pick = np.int32(pick)

    pick_width_small = idx_minus[idx_right] - idx_plus[idx_left]
    pick_value = x[pick]

    idx = np.where(x == 0)[0]
    pick_dist = pick[None, ...] - idx[..., None]
    left = pick_dist * (pick_dist > 0) + len(x) * (pick_dist < 0)
    left = idx[left.argmin(axis=0)]
    right = -pick_dist * (pick_dist < 0) + len(x) * (pick_dist > 0)
    right = idx[right.argmin(axis=0)]
    pick_width_big = right - left

    return pick, pick_value, pick_width_small, pick_width_big


def cut_trigger_from_traces(traces, center, before=.5, after=.5, dt=.00025, axis=1):
    """
    Cut trigger interval from traces
    :param traces: input seismic traces
    :param center: center of interval
    :param before: cut before, in seconds
    :param after: cut after, in seconds
    :param dt: sampling rate
    :return:
    """

    center = np.int32(center)

    _from = center - np.int32(np.round(before / dt))
    _till = center + np.int32(np.round(after / dt))

    _from = (_from >= 0) * _from + (_from < 0)
    _till = (_till < traces.shape[axis]) * _till + (_till >= traces.shape[axis]) * (traces.shape[axis] - 1)

    slc = [slice(None)] * len(traces.shape)
    slc[axis] = slice(_from, _till)

    return traces[tuple(slc)]
