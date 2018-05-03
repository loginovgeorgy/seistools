import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from .units import Units


class Horizon(object):
    def __init__(self, surface, kind='fh', name='flat'):
        self.kind = kind
        self.units = Units()
        self.name = name
        self.surface = surface

    def get_depth(self, x):
        return self.surface.get_depth(x)

    def plot(self, x=None, extent=(0, 100, 0, 100), ns=2, ax=None):
        if not self.kind == 'grid':
            if not np.any(x):
                _x, _y = np.meshgrid(
                    np.linspace(extent[0], extent[1], ns),
                    np.linspace(extent[2], extent[3], ns)
                )
                x = np.vstack((_x.ravel(), _y.ravel())).T

            z = self.surface.get_depth(x)
        else:
            x = self.surface.points[:,:-1]
            z = self.surface.points[:,-1]

        # TODO prettify using plt.show()
        if not np.any(ax):
            fig = plt.figure()
            ax = Axes3D(fig)

        self._plot_horizon_3d(x, z, ax=ax)

    @staticmethod
    def _plot_horizon_3d(x, z, ax=None):
        ax.plot_trisurf(x[:, 0], x[:, 1], np.squeeze(z))
