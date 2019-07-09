import pylab as plt
from .utils import *
from copy import deepcopy
from functools import wraps


def set_plt_params(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_font_size = plt.rcParams['font.size']
        old_serif = plt.rcParams['font.sans-serif']

        plt.rcParams['font.size'] = kwargs.get('font_size', 20)
        plt.rcParams['font.sans-serif'] = 'Arial'

        func(*args, **kwargs)

        plt.rcParams['font.size'] = old_font_size
        plt.rcParams['font.sans-serif'] = old_serif
        return

    return wrapper

# TODO add inputs checker


@set_plt_params
def plot_map(
        x,
        y,
        z,
        interp_grid_size=100,
        interp_method='linear',
        font_size=20,
        title=None,
        shift_to_min=False,
        x_label='X, m',
        y_label='Y, m',
        return_img=False,
        vmin=None,
        vmax=None,
        add_bound=True,
        add_colorbar=False,
        add_xy=False,
        colorbar_label='',
        ax=None,
        fig_width=5,
        fig_height=5,
        style='imshow',
):
    x0, y0, z0 = deepcopy(x), deepcopy(y), deepcopy(z)

    if shift_to_min:
        x0 -= x0.min()
        y0 -= y0.min()

    x, y, z = interpolate_grid(x0, y0, z0, grid_size=interp_grid_size, method=interp_method)

    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='w')

    # TODO add styles: pcolor, pcolormesh, contourf
    img = ax.imshow(
        z,
        extent=(x.min(), x.max(), y.min(), y.max()),
        cmap='seismic',
        origin='lower',
        vmin=vmin,
        vmax=vmax,
    )
    if add_colorbar:
        plot_colorbar(img, cbar_label=colorbar_label)

    if add_xy:
        ax.scatter(x0, y0, c='k')
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.set_xlabel(x_label, fontsize=font_size)

    if add_bound:
        square = get_bound_square(z)
        ax.imshow(
            square,
            alpha=1,
            extent=(x.min(), x.max(), y.min(), y.max()),
            cmap='gray',
            origin='lower',
            vmin=0,
            vmax=1
        )
    if title:
        ax.set_title(title)

    ax.grid()

    if return_img:
        return img

    return None
