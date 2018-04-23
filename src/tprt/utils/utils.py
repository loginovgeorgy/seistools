from mpl_toolkits.mplot3d import Axes3D
import pylab as plt
import numpy as np
from scipy.optimize import least_squares


def plot_points_3d(location, **kwargs):
    if not np.any(kwargs.get('ax')):
        fig = plt.figure()
        ax = Axes3D(fig)
    else:
        ax = kwargs['ax']
        kwargs.pop('ax')

    ax.scatter3D(*location, **kwargs)


def plot_line_3d(location, **kwargs):
    if not np.any(kwargs.get('ax')):
        fig = plt.figure()
        ax = Axes3D(fig)
    else:
        ax = kwargs['ax']
        kwargs.pop('ax')

    ax.plot3D(*location, **kwargs)


def is_ray_intersect_surf(sou, ray, d, top, eps=1e-8):

    def loss(r):
        (x, y, z) = sou + r * ray

        return np.squeeze((top.get_depth([x, y]) - z)**2)

    x0 = d / 2
    xs = least_squares(loss, x0, bounds=([eps], [d]))
    rec = np.array(sou + xs.x * ray, ndmin=1)
    return xs.cost < eps, rec



