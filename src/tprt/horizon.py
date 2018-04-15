import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from src.tprt.intersection_depth_calculator import FlatHorizonIDC
from .units import Units


class Horizon(object):
    def __init__(self, depth, dip, azimuth, kind='fh', name='flat', *args, **kwargs):
        self.kind = kind
        self.units = Units(**kwargs)
        self.name = name
        self.predict = None
        self._kwargs = kwargs
        self.idc = FlatHorizonIDC(depth=depth, dip=dip, azimuth=azimuth)

    def plot(self, x=None, extent=(0, 100, 0, 100), ns=10, ax=None):
        if not np.any(x):
            _x, _y = np.meshgrid(
                np.linspace(extent[0], extent[1], ns),
                np.linspace(extent[2], extent[3], ns)
            )
            x = np.vstack((_x.ravel(), _y.ravel())).T

        z = self.idc.get_depth(x)

        # TODO prettify using plt.show()
        if not np.any(ax):
            fig = plt.figure()
            ax = Axes3D(fig)

        self._plot_horizon_3d(x, z, ax=ax)

    @staticmethod
    def _plot_horizon_3d(x, z, ax=None):
        ax.plot_trisurf(x[:, 0], x[:, 1], np.squeeze(z))


