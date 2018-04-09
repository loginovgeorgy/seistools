import numpy as np
from .units import Units


def _flat_horizon(depth=0, anchor=(0, 0), dip=0, azimuth=0, **kwargs):
    # TODO: make n defined by azimuth ans dip
    n = np.float32([[0], [0], [1]])

    def plane(x):
        x[:, 0] -= anchor[0]
        x[:, 1] -= anchor[1]
        z = (depth - (x * n[:2]).sum()) / (n[2] + 1e-16)
        return z
    return plane


def _grid_horizon(**kwargs):
    return 1


HORIZON_PREDICT = {
    'flat': _flat_horizon,
    'f': _flat_horizon,
    'fh': _flat_horizon,
    'horizontal': _flat_horizon,
    'grid': _grid_horizon,

}


HORIZON_FIT = {
    'flat': None,
    'f': None,
    'fh': None,
    'horizontal': None,
    'grid': _grid_horizon,

}


class Horizon(object):
    def __init__(self, kind='fh', *args, **kwargs):
        self._predict = HORIZON_PREDICT[kind](**kwargs)
        self._fit = HORIZON_PREDICT[kind](**kwargs)
        self._args = args
        self._properties = kwargs

        self.type = type
        self.units = Units(**kwargs)

    def fit(self, x, y):
        self._fit(x, y)

    def predict(self, x, y):
        self._fit(x, y)

