from mpl_toolkits.mplot3d import Axes3D
import pylab as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import Delaunay

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




