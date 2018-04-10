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


def is_ray_intersect_surf(ray, top, eps=1e-8):

    def loss(p):
        (x, y, z) = ray.predict(*p)

        return np.squeeze((top.predict([x, y]) - z)**2)

    x0 = ray.distance/2
    xs = least_squares(loss, x0, bounds=([eps], [ray.distance]))

    return xs.cost < eps, xs.x
