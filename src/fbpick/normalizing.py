import numpy as np
from copy import deepcopy
from collections import defaultdict
from .helpers import cast_input_to_array

EPS = 1e-16

def _mean(x, axis):
    return x.mean(axis=axis, keepdims=True)


def _median(x, axis):
    return x.median(axis=axis, keepdims=True)


def _max(x, axis):
    return x.max(axis=axis, keepdims=True)


def _min(x, axis):
    return x.min(axis=axis, keepdims=True)


def _std(x, axis):
    return x.std(axis=axis, keepdims=True)


def _maxabs(x, axis):
    return np.abs(x).max(axis=axis, keepdims=True)


def _minmax(x, axis):
    return x.max(axis=axis, keepdims=True) - x.min(axis=axis, keepdims=True)


def _minmaxabs(x, axis):
    return np.abs(x).max(axis=axis, keepdims=True) - x.min(axis=axis, keepdims=True)


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


def normalize_data(x, axis=1, shift_type='mean', scale_type='maxabs', duplicate=True, cast=True,
                   calc_scale_after_shift=True):
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
    :return:
    """

    if duplicate:
        x = deepcopy(x)

    if cast:
        x = cast_input_to_array(x)

    if not calc_scale_after_shift:
        shift = calculate_shift(x, axis=axis, shift_type=shift_type)
        scale = calculate_scale(x, axis=axis, scale_type=scale_type)

        x = (x - shift) / (scale + EPS)
    else:
        x -= calculate_shift(x, axis=axis, shift_type=shift_type)
        x /= (calculate_scale(x, axis=axis, scale_type=scale_type) + EPS)

    return x




