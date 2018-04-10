import numpy as np
from .units import Units
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _flat_horizon(depth=0, anchor=(0, 0), dip=0, azimuth=0, *args, **kwargs):
    # TODO: make n defined by azimuth ans dip
    dip = np.deg2rad(dip)
    azimuth = np.deg2rad(azimuth)
    n = np.array([np.sin(dip)*np.cos(azimuth), np.sin(dip)*np.sin(azimuth)], ndmin=2)
    n3 = np.cos(dip)
    anchor = np.array(anchor, ndmin=2)

    def plane(x, *args, **kwargs):
        x = np.array(x, ndmin=2)
        x -= anchor
        z = (depth - (x * n).sum(axis=1, keepdims=True)) / (n3 + 1e-16)
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
    def __init__(self, kind='fh', name='flat', *args, **kwargs):
        self.kind = kind
        self.units = Units(**kwargs)
        self.name = name
        self.predict = None
        self._kwargs = kwargs
        self.fit(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.predict = HORIZON_FIT[self.kind](*args, **kwargs)

    def plot(self, x=None, extent=(0, 100, 0, 100), ns=10, ax=None, **kwargs):
        if not np.any(x):
            _x, _y = np.meshgrid(
                np.linspace(extent[0], extent[1], ns),
                np.linspace(extent[2], extent[3], ns)
            )
            x = np.vstack((_x.ravel(), _y.ravel())).T

        z = self.predict(x)

        # TODO prettify using plt.show()
        if not np.any(ax):
            fig = plt.figure()
            ax = Axes3D(fig)

        self._plot_horizon_3d(x, z, ax=ax)

    @staticmethod
    def _plot_horizon_3d(x, z, ax=None):
        ax.plot_trisurf(x[:, 0], x[:, 1], np.squeeze(z))


