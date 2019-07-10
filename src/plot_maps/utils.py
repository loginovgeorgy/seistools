import numpy as np
from scipy.interpolate import griddata
from copy import deepcopy
from functools import wraps


def cast_grid(x):
    x = np.array(deepcopy(x), dtype=np.float32, ndmin=1)
    return np.squeeze(x)


def decorate_input(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) == 1:
            z = args[0]
            x = kwargs.get('x', None)
            y = kwargs.get('y', None)

        elif len(args) == 3:
            x = args[0]
            y = args[1]
            z = args[2]

        else:
            raise ValueError('Input must be "(z, x=..., y=...)" or "(x, y, z)" otherwise - incorrect')
        return func(z, x=x, y=y)

    return wrapper


@decorate_input
def input_check(z, x=None, y=None):
    z = cast_grid(z)
    if len(z.shape) == 1:
        if isinstance(x, type(None)):
            x = np.arange(len(z))
        else:
            x = cast_grid(x)
            if x.shape != z.shape:
                raise ValueError(
                    "'x.shape != z.shape'. Input 'x' must be same len as 'z'. "
                    "The given 'z' was {} and the given 'x' was {}".format(
                        z.shape, x.shape
                    )
                )

        if isinstance(y, type(None)):
            y = np.arange(len(z))
        else:
            y = cast_grid(y)
            if y.shape != z.shape:
                raise ValueError(
                    "'y.shape != z.shape'. Input 'y' must be same len as 'z'. "
                    "The given 'z' was {} and the given 'y' was {}".format(
                        z.shape, y.shape
                    )
                )

    elif len(z.shape) == 2:
        if isinstance(x, type(None)):
            x = np.arange(z.shape[1])
        else:
            x = cast_grid(x)
            if x.shape != z.shape:
                raise ValueError(
                    "'x.shape != z.shape'. Input 'x' must be same shape as 'z' or same len as 'z.shape[1]'. "
                    "The given 'z' was {} and the given 'x' was {}".format(
                        z.shape, x.shape
                    )
                )

        if isinstance(y, type(None)):
            y = np.arange(z.shape[0])
        else:
            y = cast_grid(y)
            if y.shape != z.shape:
                raise ValueError(
                    "'y.shape != z.shape'. Input 'y' must be same shape as 'z' or same len as 'z.shape[0]'. "
                    "The given 'z' was {} and the given 'y' was {}".format(
                        z.shape, y.shape
                    )
                )
    else:
        raise ValueError("Input 'z' must be 1D or 2D array. The given was {}".format(z.shape))

    if (len(x.shape) == 1) & (len(y.shape) == 2):
        if len(x) != y.shape[1]:
            raise ValueError(
                    "'len(x) != y.shape[1]' .Input 'x' must be same shape as 'y' or same len as 'y.shape[1]'. "
                    "The given 'x' was {} and the given 'y' was {}".format(
                        x.shape, y.shape
                    )
            )

    if (len(x.shape) == 2) & (len(y.shape) == 1):
        if x.shape[0] != len(y):
            raise ValueError(
                    "'x.shape[0] != len(y)'. Input 'y' must be same shape as 'x' or same len as 'x.shape[0]'. "
                    "The given 'x' was {} and the given 'y' was {}".format(
                        x.shape, y.shape
                    )
            )

    return x, y, z


def interpolate_grid(z, x=None, y=None, nx=None, ny=None, method='linear'):
    """
    :param x: 1D vector
    :param y: 1D vector
    :param z: 1D vector
    :param nx:
    :param ny:
    :param method: {'linear', 'nearest', 'cubic'}
    :return:
    """
    x, y, z = input_check(x, y, z)
    if len(z.shape) == 1:
        if isinstance(nx, type(None)):
            nx = len(z)

        if isinstance(ny, type(None)):
            ny = len(z)

    if len(z.shape) == 2:
        if isinstance(nx, type(None)):
            nx = z.shape[1]

        if isinstance(ny, type(None)):
            ny = z.shape[0]

        if (len(x.shape) == 1) & (len(y.shape) == 1):
            x, y = np.meshgrid(x, y)

        if (len(x.shape) == 1) & (len(y.shape) != 1):
            x = x[None,...].repeat(z.shape[0], axis=0)

        if (len(x.shape) != 1) & (len(y.shape) == 1):
            y = y[..., None].repeat(z.shape[1], axis=1)

    xg, yg = np.meshgrid(
        np.linspace(x.min(), x.max(), nx),
        np.linspace(y.min(), y.max(), ny),
    )

    zg = griddata((x.ravel(), y.ravel()), z.ravel(), (xg, yg), method=method)

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
