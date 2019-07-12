import pylab as plt
from .utils import *

from functools import wraps
FONT_SIZE = 12


def set_plt_params(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_font_size = plt.rcParams['font.size']
        old_serif = plt.rcParams['font.sans-serif']

        plt.rcParams['font.size'] = kwargs.get('font_size', FONT_SIZE)
        plt.rcParams['font.sans-serif'] = 'Arial'

        try:
            func(*args, **kwargs)
        finally:
            plt.rcParams['font.size'] = old_font_size
            plt.rcParams['font.sans-serif'] = old_serif
        return

    return wrapper


def decorate_input_plot(func):
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

        kwargs.pop('x', None)
        kwargs.pop('y', None)
        return func(z, x=x, y=y, **kwargs)

    return wrapper


@set_plt_params
@decorate_input_plot
def plot_map(
        z,
        x=None,
        y=None,
        use_regular_grid=False,
        regular_grid_nx=100,
        regular_grid_ny=100,
        regular_grid_method='linear',
        font_size=FONT_SIZE,
        title=None,
        x_label='X, m',
        y_label='Y, m',
        alpha=1,
        vmin=None,
        vmax=None,
        add_bound=False,
        add_colorbar=False,
        colorbar_label='',
        ax=None,
        fig_width=5,
        fig_height=5,
        shift_to_min=False,
        shift_to_center=False,
        axis_off=False,
        axis_aspect_equal=True,
        axis_aspect_square=False,
        axis_grid_on=True,
        interpolation=None,
        colormap='seismic',
        edge_colors=None,
        return_img=False,
        verbose=False,
):
    """

    :param z: values 1D or 2D
    :param x: can be None, 1D or 2D
    :param y: can be None, 1D or 2D
    :param use_regular_grid: True / False
    :param regular_grid_nx: if None use z.shape[1] or len(z)
    :param regular_grid_ny: if None use z.shape[0] or len(z)
    :param regular_grid_method: one of {'linear', 'nearest', 'cubic'}
    :param font_size: 20
    :param title: 'title'
    :param x_label: 'x_label'
    :param y_label:  'y_label'
    :param alpha: None, value in range 0 - 1
    :param vmin: None, value
    :param vmax: None, value
    :param add_bound: True/False
    :param add_colorbar: True/False
    :param colorbar_label: 'Title'
    :param ax: None or Axes object
    :param fig_width: units
    :param fig_height: units
    :param shift_to_min: True/False
    :param shift_to_center: True/False
    :param axis_off: True/False
    :param axis_aspect_equal: True/False
    :param axis_aspect_square: True/False
    :param axis_grid_on: True/False
    :param interpolation: one of: {'none', 'nearest', 'bilinear', 'bicubic',
    'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
    'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'}

    :param colormap:
    :param edge_colors: None or color: 'k', 'r', 'b', 'g'
    :param return_img:
    :param verbose:
    :return:
    """
    if use_regular_grid | (not isinstance(interpolation, type(None))):
        x, y, z = interpolate_grid(
            z,
            x=x,
            y=y,
            nx=regular_grid_nx,
            ny=regular_grid_ny,
            method=regular_grid_method,
            verbose=verbose,
        )
    else:
        x, y, z = input_check(x, y, z)

    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='w')

    if isinstance(edge_colors, type(None)):
        edge_colors = str(edge_colors)

    if shift_to_min:
        x -= np.nanmin(x)
        y -= np.nanmin(y)

    if shift_to_center:
        x -= np.nanmean(x)
        y -= np.nanmean(y)

    x_lim = np.array([x.min(), x.max()])
    y_lim = np.array([y.min(), y.max()])

    plot_func = ax.pcolormesh
    plot_func_args = [x, y, z]
    plot_func_kwargs = dict(
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        edgecolors=edge_colors,
    )

    if (len(z.shape) == 1) & (not use_regular_grid) & isinstance(interpolation, type(None)):
        plot_func = ax.tripcolor
        add_bound = False

    if not isinstance(interpolation, type(None)):

        plot_func = ax.imshow
        plot_func_args = [z]
        plot_func_kwargs = dict(
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            extent=(x.min(), x.max(), y.min(), y.max()),
            origin='lower',
            interpolation=interpolation,
        )

    img = plot_func(
        *plot_func_args,
        **plot_func_kwargs,
    )

    if add_colorbar:
        plot_colorbar(img, cbar_label=colorbar_label)

    # TODO prettify plotting of the boundary
    if add_bound:
        square = get_bound_square(z)
        ax.pcolormesh(
            x,
            y,
            square,
            cmap='gray',
        )

    ax.set_ylabel(y_label, fontsize=font_size)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if title:
        ax.set_title(title)

    if axis_off:
        ax.set_axis_off()

    if axis_aspect_equal:
        ax.set_aspect('equal')

    if axis_aspect_square:
        ax.set_aspect('equal')
    if axis_grid_on:
        ax.grid()

    if return_img:
        return img

    return None
