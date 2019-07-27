import pylab as plt
import numpy as np
from .seismic_wiggle import wiggle
from .normalizing import normalize_data
from scipy.interpolate import griddata

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

MARKERS = ['s', 'D', 'd', 'o', '.', 'x', '+']


def _colorbar(mappable, cbar_label=''):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.04)
    cbar = fig.colorbar(mappable, cax=cax, fraction=0.022)
    cbar.set_label(cbar_label, rotation=270)
    return cbar


def plot_shot(traces, fr, picks=None, figsize=(10, 5), gain=500, clip=.5, ylim=None, xticklabel=None, profile=True, **kwargs):
    to = fr[1]
    fr = fr[0]
    temp = traces[fr:to].T.copy()
    temp /= np.abs(temp).max() / 2
    if picks:
        pks = {x: picks[x][fr:to].copy() for x in picks}
    else:
        pks = None
    if np.any(np.array(xticklabel)):
        xticklabel = np.array(xticklabel)[fr:to]

    wiggle(temp,
           picks=pks,
           gain=gain,
           clip=clip,
           figsize=figsize,
           curve=profile,
           ylim=ylim,
           xticklabel=xticklabel,
           **kwargs)


def interactive_plot_shot(
        traces,
        figsize=(30, 10),
        initial_xlim=(0, 50),
        initial_ylim=(0, 100),
        fontsize=15,
        xticklabel=None,
        **kwargs,
):
    plt.rcParams.update({'font.size': fontsize})
    fr = widgets.IntRangeSlider(
        min=0, max=traces.shape[0],
        step=5, value=[max([initial_xlim[0], 0]), min([traces.shape[0], initial_xlim[1]])], continuous_update=False
    )
    gain = widgets.FloatSlider(min=.0000001, max=2000, step=1, value=1, continuous_update=False)
    clip = fixed(1)
    ylim = widgets.IntRangeSlider(
        min=0, max=traces.shape[1], step=5, value=[max([0, initial_ylim[0]]), min([traces.shape[1], initial_ylim[1]])],
        continuous_update=False
    )

    traces = fixed(traces)
    figsize = fixed(figsize)

    def plot_traces(picks):
        interact(
            plot_shot,
            traces=traces,
            picks=fixed(picks),
            figsize=figsize,
            fr=fr,
            gain=gain,
            clip=clip,
            ylim=ylim,
            xticklabel=fixed(xticklabel),
            **kwargs,
        )

    return plot_traces


def interpolate_map(x, y, z, grid_size=100):
    x = np.float32(x)
    y = np.float32(y)
    z = np.float32(z)
    xg, yg = np.meshgrid(
        np.linspace(x.min(), x.max(), grid_size),
        np.linspace(y.min(), y.max(), grid_size),
    )
    zg = griddata((x, y), z, (xg, yg), method='linear')

    return xg, yg, zg


def get_bound_square(x, y, z):
    _z = z.copy().T
    _z[~np.isnan(_z)] = 0
    _z[np.isnan(_z)] = 1
    _z = np.abs(np.gradient(_z.T, axis=0))
    _z[_z == 0] = np.nan

    return np.float32(_z)


def plot_map(
        values,
        grid_size=100,
        colorbar=False,
        fontsize=12,
        title=None,
        ax=None,
        figsize=(10, 10),
        shift=True,
        xlabel='X, m',
        ylabel='Y, m',
        return_img=False,
        vmin=None,
        vmax=None,
        cbar_label='',
):
    plt.rcParams.update({'font.size': fontsize})
    x, y, z = interpolate_map(*values, grid_size=grid_size)

    if shift:
        x -= x.min()
        y -= y.min()

    if not ax:
        fig, ax = plt.subplots(figsize=figsize, facecolor='w')

    img = ax.imshow(
        z,
        extent=(x.min(), x.max(), y.min(), y.max()),
        cmap='seismic',
        origin='lower',
        vmin=vmin,
        vmax=vmax,
    )
    if colorbar:
        _colorbar(img, cbar_label=cbar_label)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    square = get_bound_square(x, y, z)
    ax.imshow(square, alpha=1, extent=(x.min(), x.max(), y.min(), y.max()), cmap='gray', origin='lower')
    if title:
        ax.set_title(title)
    if return_img:
        return img


def plot_tau_offset(df,
                    column_offset='OFFSET',
                    columns_tau='FB',
                    dt=1,
                    grid=True,
                    markersize=1,
                    fontsize=20,
                    ylim=None,
                    title=None,
                    ax=None,
                    figsize=(10, 6),
                    xlabel='Offset, m',
                    ylabel='Tau, s',
                    divide_dt=True,
                    ):
    plt.rcParams.update({'font.size': fontsize})

    if not isinstance(columns_tau, list):
        columns_tau = [columns_tau]

    columns_tau = columns_tau[:len(MARKERS)]
    use_cols = [column_offset] + columns_tau
    df_cur = df[use_cols].copy()
    if divide_dt:
        df_cur[columns_tau] /= dt
        df_cur[columns_tau] = df_cur[columns_tau].round(0)
    df_cur = df_cur[use_cols].drop_duplicates()
    df_cur[columns_tau] *= dt
    color = {c: plt.cm.tab10(i) for i, c in enumerate(columns_tau)}
    marker = {c: MARKERS[i] for i, c in enumerate(columns_tau)}

    if not ax:
        fig, ax = plt.subplots(figsize=figsize, facecolor='w')

    for c in columns_tau:
        params = dict(
            marker=marker[c],
            markerfacecolor=color[c],
            markeredgecolor=color[c],
            linestyle='None',
            alpha=.5,
        )
        ax.plot(
            df_cur[column_offset].values,
            df_cur[c].values,
            markersize=markersize,
            **params,
        )
        ax.plot(np.nan, np.nan,
                 label=c,
                 markersize=10,
                 **params,
                 )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    if len(columns_tau) > 1:
        ax.legend(fontsize=fontsize, loc=2)

    if grid:
        ax.grid()
    if title:
        ax.set_title(title)


def select_offset_bin(df, tag, columns_hist, xlim=(0, 1.5), ylim=(-1, 5), bins=50):
    df_cur = df.set_index('OFFSET_BIN').loc[tag].copy()
    plot_tau_hist(df_cur, columns_hist, xlim=xlim, ylim=ylim, bins=bins)


def plot_tau_hist(df_cur, columns_hist, xlim=(0, 1.5), ylim=(-1, 5), bins=50, fontsize=20):
    plt.rcParams.update({'font.size': fontsize})

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='w')

    mean_vals = df_cur[columns_hist].mean()
    median_vals = df_cur[columns_hist].median()
    std_vals = df_cur[columns_hist].std(ddof=0)

    color = {c: plt.cm.tab10(i) for i, c in enumerate(columns_hist)}
    for c in columns_hist:
        df_cur[c].plot.hist(ax=ax, bins=bins, bottom=0.1, alpha=.5, color=color[c])
        ax.plot(mean_vals[c] * np.ones(2), np.array([.1, 10 ** 5]), label=c + '_mean', color=color[c])
        ax.plot(median_vals[c] * np.ones(2), np.array([.1, 10 ** 5]), label=c + '_median', color=color[c],
                linestyle='dashed')
        for s in range(3, 6):
            ax.plot(
                median_vals[c] + std_vals[c] * s * np.ones(2),
                np.array([.1, 10 ** 5]),
                color=color[c],
                linestyle='dotted'
            )

            ax.plot(
                median_vals[c] - std_vals[c] * s * np.ones(2),
                np.array([.1, 10 ** 5]),
                color=color[c],
                linestyle='dotted'
            )

    #             ax.text(median_vals[c]+ std_vals[c]*s, 1, str(s), color=color[c], fontsize=20)
    #             ax.text(median_vals[c]- std_vals[c]*s, 1, '-' + str(s), color=color[c], fontsize=20)

    ax.legend(fontsize=15, loc=2)
    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim([10 ** ylim[0], 10 ** ylim[1]])
    ax.set_xlabel('Tau, s', fontsize=20)
    ax.set_ylabel('log count', fontsize=20)


def palette_tau_hist(
        df,
        columns_hist,
        index_col='OFFSET_BIN',
        tau_lim=(0, 1.6),
        bins=100,
        figsize=(20, 10),
        fontsize=20,
        bin_label='mid',
        dt=1,
):
    def _bin_label(bin, style):
        if style == 'full':
            return str(bin)
        if style == 'mid':
            return str(np.round(bin.mid))
        if style == 'left':
            return str(np.round(bin.left))
        if style == 'right':
            return str(np.round(bin.right))

    plt.rcParams.update({'font.size': fontsize})
    if not isinstance(columns_hist, list):
        columns_hist = [columns_hist]

    index = df[index_col].unique()
    index = np.sort(index)
    index = index[::-1]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, facecolor='w')
    color = {c: plt.cm.tab10(i) for i, c in enumerate(columns_hist)}

    histograms = {}
    v_max = 0
    for _tag in index:
        df_cur = df.set_index(index_col).loc[_tag].copy()

        tag = _bin_label(_tag, bin_label)
        histograms[tag] = {}
        for c in columns_hist:
            histograms[tag][c] = {}

            vals = df_cur[c].values / dt
            hist, bin_edges = np.histogram(vals/dt, range=tau_lim, bins=bins)
            hist = np.log10(hist + .1)
            de = np.diff(bin_edges)[0]
            bin_edges = bin_edges[:-1][hist > 0]
            hist = hist[hist > 0]

            histograms[tag][c]['hist'] = hist
            histograms[tag][c]['bin_edges'] = bin_edges
            histograms[tag][c]['de'] = de

            histograms[tag][c]['mean'] = np.nanmean(vals)
            histograms[tag][c]['median'] = np.nanmedian(vals)
            histograms[tag][c]['std'] = np.nanstd(vals, ddof=0)

            v_max = max([v_max, hist.max()])

    v_max = np.round(v_max) + 1
    value_ticks = np.arange(len(index) * v_max)
    value_tick_labels = [str(x) for x in np.arange(np.int32(v_max))] * len(index)
    value_lim = [0, len(index) * v_max]

    for i, _tag in enumerate(index):

        tag = _bin_label(_tag, bin_label)
        shift = i * v_max

        ax.text(i * v_max + v_max / 2, tau_lim[1], tag, fontsize=fontsize, horizontalalignment='center',
                verticalalignment='top')
        for c in columns_hist:
            if i > 0:
                param = [{}, {}]
            else:
                param = [{'label': c + '_mean'}, {'label': c + '_median'}]

            width = histograms[tag][c]['hist']
            y = histograms[tag][c]['bin_edges']
            height = histograms[tag][c]['de']

            ax.barh(
                y=y,
                width=width,
                height=height,
                left=shift,
                color=color[c],
                alpha=.5,
            )

            ax.plot([shift, shift], tau_lim, 'k')

            mean_val = histograms[tag][c]['mean']
            median_val = histograms[tag][c]['median']
            std_val = histograms[tag][c]['std']

            ax.plot(
                [i * v_max, (i + 1) * v_max],
                mean_val * np.ones(2),
                color=color[c],
                **param[0],
            )
            ax.plot(
                [i * v_max, (i + 1) * v_max],
                median_val * np.ones(2),
                color=color[c],
                linestyle='dashed',
                **param[1],
            )

            for s in range(3, 6):
                ax.plot(
                    [i * v_max, (i + 1) * v_max],
                    median_val + std_val * s * np.ones(2),
                    color=color[c],
                    linestyle='dotted',
                )

                ax.plot(
                    [i * v_max, (i + 1) * v_max],
                    median_val - std_val * s * np.ones(2),
                    color=color[c],
                    linestyle='dotted',
                )

    ax.legend(loc=0)
    ax.set_xticks(value_ticks)
    ax.set_xticklabels(value_tick_labels)
    ax.set_xlim(value_lim)
    ax.set_xlabel('Count, log10(N)')

    ax.set_ylim(tau_lim)
    ax.set_ylabel('Tau, s')
    ax.grid(True)


def palette_tau_hist_vertical(
        df,
        columns_hist,
        index_col='OFFSET_BIN',
        tau_lim=(0, 1.6),
        bins=100,
        figsize=(10, 20),
        fontsize=20,
        bin_label='mid',
        dt=1,
        ytick_freq=2,
        min_sigma=3,
        max_sigma=5,
        hold_v_max=None,
):
    def _bin_label(bin, style):
        if style == 'full':
            return str(bin)
        if style == 'mid':
            return str(np.round(bin.mid))
        if style == 'left':
            return str(np.round(bin.left))
        if style == 'right':
            return str(np.round(bin.right))

    plt.rcParams.update({'font.size': fontsize})
    if not isinstance(columns_hist, list):
        columns_hist = [columns_hist]

    index = df[index_col].unique()
    index = np.sort(index)
    index = index[::-1]

    histograms = {}
    v_max = 0
    for _tag in index:
        df_cur = df.set_index(index_col).loc[_tag].copy()

        tag = _bin_label(_tag, bin_label)
        histograms[tag] = {}
        for c in columns_hist:
            histograms[tag][c] = {}

            vals = df_cur[c].values / dt
            hist, bin_edges = np.histogram(vals / dt, range=tau_lim, bins=bins)
            hist = np.log10(hist + .1)
            de = np.diff(bin_edges)[0]
            bin_edges = bin_edges[:-1][hist > 0]
            hist = hist[hist > 0]

            histograms[tag][c]['hist'] = hist
            histograms[tag][c]['bin_edges'] = bin_edges
            histograms[tag][c]['de'] = de

            histograms[tag][c]['mean'] = np.nanmean(vals)
            histograms[tag][c]['median'] = np.nanmedian(vals)
            histograms[tag][c]['std'] = np.nanstd(vals, ddof=0)

            v_max = max([v_max, hist.max()])
    if hold_v_max:
        v_max = hold_v_max
    v_max = np.round(v_max) + 1
    if np.mod(v_max, 2):
        v_max += 1

    value_ticks = np.arange(0, len(index) * v_max, ytick_freq)
    value_tick_labels = np.arange(0, v_max, ytick_freq)
    value_tick_labels = [str(x) for x in value_tick_labels] * len(index)
    value_lim = [0, len(index) * v_max]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, facecolor='w')
    color = {c: plt.cm.tab10(i) for i, c in enumerate(columns_hist)}

    for i, _tag in enumerate(index):

        tag = _bin_label(_tag, bin_label)
        shift = i * v_max

        ax.text(tau_lim[1], i * v_max + v_max / 2, tag,
                fontsize=fontsize,
                horizontalalignment='right',
                verticalalignment='center'
                )

        for c in columns_hist:
            if i > 0:
                param = [{}, {}]
            else:
                if len(columns_hist) > 1:
                    param = [{'label': c + '_mean'}, {'label': c + '_median'}]
                else:
                    param = [{'label': 'mean'}, {'label': 'median'}]

            height = histograms[tag][c]['hist']
            x = histograms[tag][c]['bin_edges']
            width = histograms[tag][c]['de']

            ax.bar(
                x=x,
                width=width,
                height=height,
                bottom=shift,
                color=color[c],
                alpha=.5,
            )

            ax.plot(tau_lim, [shift, shift], 'k')

            mean_val = histograms[tag][c]['mean']
            median_val = histograms[tag][c]['median']
            std_val = histograms[tag][c]['std']

            ax.plot(
                mean_val * np.ones(2),
                [i * v_max, (i + 1) * v_max],
                color=color[c],
                **param[0],
            )
            ax.plot(
                median_val * np.ones(2),
                [i * v_max, (i + 1) * v_max],
                color=color[c],
                linestyle='dashed',
                **param[1],
            )

            for s in range(min_sigma, max([min_sigma, max_sigma]) + 1):
                ax.plot(
                    median_val + std_val * s * np.ones(2),
                    [i * v_max, (i + 1) * v_max],
                    color=color[c],
                    linestyle='dotted',
                )

                ax.plot(
                    median_val - std_val * s * np.ones(2),
                    [i * v_max, (i + 1) * v_max],
                    color=color[c],
                    linestyle='dotted',
                )

    ax.legend(loc=0)
    ax.set_yticks(value_ticks)
    ax.set_yticklabels(value_tick_labels)
    ax.set_ylim(value_lim)
    ax.set_ylabel('Count, log10(N)')

    ax.set_xlim(tau_lim)
    ax.set_xlabel('Tau, s')
    ax.grid(True)

###################################### Trash
def plot_3c_data(rec, ax, j):
    col = {0: 'r', 1: 'g', 2: 'b', 3: 'k', 4: 'm'}
    for i in range(rec.shape[-1]):
        tr = rec[:, i]
        tr = tr - tr.mean()
        tr = tr / (np.abs(rec).max() + 2e-15) / 2
        ax.plot(tr + j, col[i])


def plot_peaks(axs, mask, n):
    col = {0: 'r', 1: 'g', 2: 'b', 3: 'k', 4: 'm'}
    if len(mask.shape) < 2:
        mask = np.expand_dims(mask, axis=-1)
    for ax in axs:
        for i in range(mask.shape[-1]):
            for jrec in range(mask.shape[0]):
                ax.plot(mask[jrec, i] * np.array([1, 1]), jrec + np.array([-.3, .3]), color=plt.cm.Dark2(i))

                if jrec > n:
                    break


def plot_pallete(fr, t, x, maped, kf=10, ncut=16):
    if not isinstance(maped, list):
        maped = [maped]

    ns = x.shape[1]
    nr = x.shape[1]

    plt.figure(figsize=(kf, kf * np.int32(nr / ns)))
    plt.imshow(np.squeeze(x)[fr:t].T, 'seismic')
    for i, mt in enumerate(maped):
        mt = mt.copy()
        if len(mt.shape) > 1:
            mt[:, :ncut] = 0
            mt[:, -ncut:] = 0
            mt = mt.argmax(axis=1)
        plt.plot(mt[fr:t], color=plt.cm.tab10(i))
    ax = plt.gca()
    ax.set_ylim([0, ns])
    plt.show()


def signals_dashboard(*args, n=10, mask=None, **kwargs):
    f, axs = plt.subplots(1, len(args), figsize=(10 * (len(args) + 1), 10))

    for i, x in enumerate(args):
        if len(x.shape) < 3:
            x = np.expand_dims(x, axis=-1)

        for jrec in range(x.shape[0]):
            rec = x[jrec]
            plot_3c_data(rec, axs[i], jrec)

            if jrec > n:
                break
        axs[i].set_xlim([0, len(rec)])
    if np.any(mask):
        plot_peaks(axs, mask, n)

    plt.show()


def plot_sdata(x, true=None, pred=None, nr=10, rand=False):
    nr = np.min([x.shape[0], nr])
    if rand:
        recs = np.random.randint(0, x.shape[0], nr)
    #         recs = np.arange(0,nr)
    else:
        recs = np.arange(nr)
    x = x[recs, :]
    if true is not None:
        true = true[recs, :]

    if pred is not None:
        pred = pred[recs, :]

    for i, s in enumerate(x):
        plt.plot(s + i, 'k')

        if true is not None:
            j = true[i, :].argmax() - 1
            plt.plot(j, s[j] + i, 'sr')

        if pred is not None:
            j = pred[i, :].argmax() - 1
            plt.plot(j, s[j] + i, '^b')

    plt.show()


def plot_seism(x):
    for i in range(x.shape[1]):
        plt.plot(x[:, i, 0] + i, 'r')
        plt.plot(x[:, i, 1] + i, 'g')
        plt.plot(x[:, i, 2] + i, 'b')


def plot_sdata_jy(x, true=None, pred=None, nr=10, rand=False):
    nr = np.min([x.shape[0], nr])
    if rand:
        recs = np.random.randint(0, x.shape[0], nr)
    else:
        recs = np.arange(nr)
    x = x[recs, :]
    if true is not None:
        true = true[recs]

    if pred is not None:
        pred = pred[recs]

    for i, s in enumerate(x):
        plt.plot(s + i, 'k')

        if true is not None:
            j = true[i].astype(np.int32)
            plt.plot(j, s[j] + i, 'sr')

        if pred is not None:
            j = pred[i].astype(np.int32)
            plt.plot(j, s[j] + i, '^b')

    plt.show()


def plot_spec(abs_s):
    for s in abs_s:
        plt.plot(s)

    plt.show()


def plot_3c(x, img, picks=None, w=10, h=20, dt=1, title=''):
    if not isinstance(img, list):
        img = [img]

    if not picks:
        picks = {}

    fig, axs = plt.subplots(ncols=len(img), figsize=(w * len(img), h), squeeze=False)
    axs = axs[0]

    if len(x.shape) == 3:
        nc = x.shape[2]

    if len(x.shape) == 2:
        x = x[..., None]
        nc = 1
    #     return axs[0]
    #     return np.squeeze(axs)
    #     axs = np.ar(axs)
    #         ax1, ax2 = axs
    color = ['r', 'g', 'b']
    frm = -.5
    tll = x.shape[0] * nc - .5

    kwargs = dict(
        aspect='auto',
        origin='lower',
        extent=(0, x.shape[1] * dt, frm, tll),
        cmap=plt.cm.Greys,
        vmin=0,
        vmax=1,
    )
    x = normalize_data(x, axis=1, shift_type='mean', scale_type='maxabs')
    for i in range(x.shape[0]):
        for c in range(nc):
            for ax in axs:
                ax.plot(np.arange(len(x[i, :, c])) * dt, x[i, :, c] / 2 + (i * nc) + ((nc - 1) - c), color=color[c])

        for ax in axs:
            ax.plot([0, x.shape[1] * dt], np.zeros(2) + i * nc + 2.5, color='k', linestyle='-.', linewidth=.5)
            for ip, key in enumerate(picks):
                ax.plot(picks[key][i] * np.ones(2) * dt, [i * nc - .5, i * nc + 2.5], color=plt.cm.tab10(ip),
                        linestyle='--')

    for ip, key in enumerate(picks):
        axs[0].plot(np.nan, np.nan, color=plt.cm.tab10(ip), linestyle='--', label=key)
    if picks:
        axs[0].legend()
    for ax, _img in zip(axs, img):
        ax.imshow(_img, **kwargs)
        #         ax.set_yticks(np.arange(1, x.shape[0]*3 - 1, 3))
        #         ax.set_yticklabels(np.arange(1, x.shape[0] + 1, 1))
        #         ax.grid(axis='y')

        ax.set_title(title)
        ax.set_xlim([0, x.shape[1] * dt])
        ax.set_ylim([frm, tll])