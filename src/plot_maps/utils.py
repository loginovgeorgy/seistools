import numpy as np
from scipy.interpolate import griddata


def interpolate_grid(x, y, z, grid_size=100, method='linear'):
    """

    :param x: 1D vector
    :param y: 1D vector
    :param z: 1D vector
    :param grid_size:
    :param method: {'linear', 'nearest', 'cubic'}
    :return:
    """
    x = np.squeeze(np.float32(x))
    y = np.squeeze(np.float32(y))
    z = np.squeeze(np.float32(z))
    xg, yg = np.meshgrid(
        np.linspace(x.min(), x.max(), grid_size),
        np.linspace(y.min(), y.max(), grid_size),
    )
    zg = griddata((x, y), z, (xg, yg), method=method)

    return xg, yg, zg


def get_bound_square(z):
    _z = z.copy().T
    _z[~np.isnan(_z)] = 0.
    _z[np.isnan(_z)] = 1.
    _z = np.abs(np.gradient(_z.T, axis=0))
    _z[_z == 0] = np.nan

    return np.float32(_z)


def plot_colorbar(mappable, cbar_label=''):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.04)
    cbar = fig.colorbar(mappable, cax=cax, fraction=0.022)
    cbar.set_label(cbar_label, rotation=270)
    return cbar
