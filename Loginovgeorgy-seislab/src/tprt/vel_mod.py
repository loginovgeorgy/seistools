import numpy as np
from .units import Units


def _flat_horizon(*args, depth=0, anchor=(0, 0), dip=0, azimuth=0, **kwargs):
    # TODO: make n defined by azimuth ans dip
    n = np.float32([[0], [0], [1]])

    def plane(x, *args, **kwargs):
        x[:, 0] -= anchor[0]
        x[:, 1] -= anchor[1]
        z = (depth - (x * n[:2]).sum()) / (n[2] + 1e-16)
        return z
    return plane


def _grid_horizon(**kwargs):
    return 1


HORIZON_FIT = {
    'flat': _flat_horizon,
    'f': _flat_horizon,
    'fh': _flat_horizon,
    'horizontal': _flat_horizon,
    'grid': _grid_horizon,

}


class Horizon(object):
    def __init__(self,  *args, kind='fh', name='flat', **kwargs):
        self.predict = None
        self._fit = HORIZON_FIT[kind](**kwargs)
        self._args = args
        self._kwargs = kwargs

        self.kind = kind
        self.units = Units(**kwargs)
        self.name = name

    def fit(self, x, y):
        self.predict = self._fit(x, y)


def _iso_model(*args, vp=3500, vs=3500, **kwargs):
    def velocity(*args):
        return {'vp': vp, 'vs': vs}
    return velocity


LAYER_FIT = {
    'iso': _iso_model,
    'ani': None,
}


class Layer(object):
    def __init__(self, *args, kind='iso', name='flat', **kwargs):
        self.fit = LAYER_FIT[kind](*args, **kwargs)
        self.top = Horizon(**kwargs)
        self.units = Units(**kwargs)
        self.name = name
