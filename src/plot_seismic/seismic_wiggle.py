import numpy as np
import matplotlib.pyplot as plt
from .helpers import insert_zeros_in_trace, input_check, input_check_color_dicts

MARKERS = ['s', 'D', 'd', 'o', '.', 'x', '+']

COLORMAP = plt.cm.tab10


def plot_traces(
        traces,
        dt=1.,
        picks=None,
        offset=None,
        mask=None,
        gain=1.,
        clip=None,
        max_norm=True,
        start_time=0,
        time_label='time',
        traces_label='trace',
        title='',
        time_vertical=False,
        fill_positive=False,
        fill_negative=False,
        trace_color=None,
        invert_y_axis=True,
        picks_on_amplitude=False,
        picks_curve=False,
        alpha=.5,
        dist_for_3c=.5,
        font_size=20,
        ax=None,
        fig_width=10,
        fig_height=10,
):
    old_font_size = plt.rcParams['font.size']
    old_serif = plt.rcParams['font.sans-serif']
    plt.rcParams['font.size'] = font_size
    plt.rcParams['font.sans-serif'] = 'Arial'

    traces, offset, mask, picks = input_check(traces, offset, mask, picks)
    offset_ticks = offset.copy()
    offset = np.arange(traces.shape[0])

    trace_color, fill_positive, fill_negative = input_check_color_dicts(
        traces.shape[2],
        **dict(
            trace_color=trace_color,
            fill_positive=fill_positive,
            fill_negative=fill_negative,
        )
    )

    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='w')

    if max_norm:
        traces /= np.nanmax(np.abs(traces) * 2)

    traces *= np.float32(np.abs(gain))

    if clip:
        traces[traces > clip] = clip
        traces[traces < -clip] = -clip

    def _get_x_y(_x):
        return _x

    _fill = ax.fill_between
    _set_time_label = ax.set_xlabel
    _set_time_lim = ax.set_xlim
    _set_traces_lim = ax.set_ylim
    _set_traces_label = ax.set_ylabel
    _set_trace_ticks = ax.set_yticks
    _set_trace_tick_labels = ax.set_yticklabels
    _set_grid_axis = 'y'

    k = np.float32(traces.shape[2] * dist_for_3c) + 1e-15
    traces_lim = np.array([-.5, traces.shape[0] - .5], dtype=np.float32)
    traces_lim += np.array([- np.nanmax(np.abs(traces[0])), np.nanmax(np.abs(traces[-1]))]) / k

    time_lim = np.array([0, traces.shape[1]]) * dt + start_time

    if invert_y_axis:
        fill_positive, fill_negative = fill_negative, fill_positive
        dist_for_3c *= -1.

    if time_vertical:
        def _get_x_y(_x): return _x[::-1]

        _fill = ax.fill_betweenx
        _set_time_label = ax.set_ylabel
        _set_time_lim = ax.set_ylim
        _set_traces_lim = ax.set_xlim
        _set_traces_label = ax.set_xlabel
        _set_trace_ticks = ax.set_yticks
        _set_trace_tick_labels = ax.ax.set_xticklabels
        _set_grid_axis = 'x'

    time = np.arange(traces.shape[1]) * dt

    for jt in range(traces.shape[0]):
        off = offset[jt]

        for jc in range(traces.shape[2]):
            trace = traces[jt, :, jc]
            if any(fill_positive) | any(fill_negative):
                trace, time = insert_zeros_in_trace(trace)

            trace /= (traces.shape[2] * dist_for_3c)
            shift = .5 + (jc - traces.shape[2]) / (traces.shape[2] + 1)
            shift *= dist_for_3c
            shift += off

            if any(fill_positive):
                _fill(
                    time + start_time,
                    shift,
                    trace + shift,
                    where=trace >= 0,
                    alpha=alpha,
                    facecolor=fill_positive[jc],
                )

            if any(fill_negative):
                _fill(
                    time + start_time,
                    shift,
                    trace + shift,
                    alpha=alpha,
                    where=trace <= 0,
                    facecolor=fill_negative[jc],
                )
            x, y = _get_x_y([time + start_time, trace + shift])
            ax.plot(x, y, color=trace_color[jc])

            for ip, label in enumerate(picks):
                p = picks[label]
                pick_time = p[jt] * dt + start_time
                if traces.shape[2] == 1:
                    pick_amplitude = off + trace[np.int32(p[jt])] * picks_on_amplitude
                    marker = MARKERS[ip]
                    line_style = 'None'
                else:
                    pick_amplitude = [off - .3, off + .3]
                    pick_time = [pick_time, pick_time]
                    marker = None
                    line_style = 'dashed'

                x, y = _get_x_y([pick_time, pick_amplitude])
                ax.plot(x,
                        y,
                        color=COLORMAP(ip),
                        markeredgecolor=COLORMAP(ip),
                        markerfacecolor='None',
                        marker=marker,
                        linestyle=line_style,
                        )

    if picks:
        j_traces = np.arange(traces.shape[0])
        for ip, label in enumerate(picks):
            if picks_curve:
                p = np.int32(picks[label])
                pick_times = p * dt + start_time

                if traces.shape[2] == 1:
                    pick_amplitudes = offset + traces[j_traces, p, 0] * picks_on_amplitude
                    marker = MARKERS[ip]
                else:
                    pick_amplitudes = np.insert(offset - .5, np.arange(len(offset))+1, offset + .5)
                    pick_times = np.insert(pick_times, np.arange(len(offset))+1, pick_times)
                    marker = None

                x, y = _get_x_y([pick_times, pick_amplitudes])
                ax.plot(
                    x,
                    y,
                    color=COLORMAP(ip),
                    label=label,
                    markeredgecolor=COLORMAP(ip),
                    markerfacecolor='None',
                    marker=marker,
                    linestyle='solid',
                )
            else:
                ax.plot(
                    np.nan,
                    np.nan,
                    color=COLORMAP(ip),
                    label=label,
                    markeredgecolor=COLORMAP(ip),
                    markerfacecolor='None',
                    marker=MARKERS[ip],
                    linestyle='None'
                )

    if not isinstance(mask, type(None)):
        c_map = plt.cm.Greys
        # c_map.set_bad('green', 1.)
        extent = (time_lim[0], time_lim[1], -.5, traces.shape[0] - .5)

        mask /= np.nanmax(np.abs(mask))
        if time_vertical:
            extent = (time_lim[1], -.5, traces.shape[0] - .5, time_lim[0], time_lim[1])
            mask = mask.T

        kwargs = dict(
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap=c_map,
            vmin=0,
            vmax=1.5,
        )

        ax.imshow(mask, **kwargs)

    _set_time_label(time_label)
    _set_time_lim(time_lim)
    _set_traces_label(traces_label)
    _set_traces_lim(traces_lim)
    _set_trace_ticks(np.arange(traces.shape[0]))
    _set_trace_ticks(np.arange(traces.shape[0] - 1) + .5, minor=True)
    _set_trace_tick_labels(offset_ticks)

    ax.grid(which='minor', axis=_set_grid_axis)
    ax.set_title(title)

    # TODO: handle with timestamps

    # ax.set_xticks(np.arange(0, data.shape[0], 5))
    # if np.any(np.array(xticklabel)):
    #     ticks = np.array(ax.get_xticks(), dtype=int)
    #     ticks = ticks[ticks < len(xticklabel)]
    #     xticklabel = np.array(xticklabel)[ticks]
    #     ax.set_xticklabels(xticklabel)

    if invert_y_axis:
        ax.invert_yaxis()

    plt.rcParams['font.size'] = old_font_size
    plt.rcParams['font.sans-serif'] = old_serif

# def plot_3c(x, img=None, picks=None, w=10, h=20, dt=1, title=''):
#     if not isinstance(img, list):
#         img = [img]
#
#     if not picks:
#         picks = {}
#
#     fig, axs = plt.subplots(ncols=len(img), figsize=(w * len(img), h), squeeze=False)
#     axs = axs[0]
#
#     if len(x.shape) == 3:
#         nc = x.shape[2]
#
#     if len(x.shape) == 2:
#         x = x[..., None]
#         nc = 1
#     #     return axs[0]
#     #     return np.squeeze(axs)
#     #     axs = np.ar(axs)
#     #         ax1, ax2 = axs
#     color = ['r', 'g', 'b']
#     frm = -.5
#     tll = x.shape[0] * nc - .5
#
#     kwargs = dict(
#         aspect='auto',
#         origin='lower',
#         extent=(0, x.shape[1] * dt, frm, tll),
#         cmap=plt.cm.Greys,
#         vmin=0,
#         vmax=1,
#     )
#     x = normalize_data(x, axis=1, shift_type='mean', scale_type='maxabs')
#     for i in range(x.shape[0]):
#         for c in range(nc):
#             for ax in axs:
#                 ax.plot(np.arange(len(x[i, :, c])) * dt, x[i, :, c] / 2 + (i * nc) + ((nc - 1) - c), color=color[c])
#
#         for ax in axs:
#             ax.plot([0, x.shape[1] * dt], np.zeros(2) + i * nc + 2.5, color='k', linestyle='-.', linewidth=.5)
#             for ip, key in enumerate(picks):
#                 ax.plot(picks[key][i] * np.ones(2) * dt, [i * nc - .5, i * nc + 2.5], color=plt.cm.tab10(ip),
#                         linestyle='--')
#
#     for ip, key in enumerate(picks):
#         axs[0].plot(np.nan, np.nan, color=plt.cm.tab10(ip), linestyle='--', label=key)
#     if picks:
#         axs[0].legend()
#     for ax, _img in zip(axs, img):
#         if not isinstance(_img, type(None)):
#             ax.imshow(_img, **kwargs)
#         #         ax.set_yticks(np.arange(1, x.shape[0]*3 - 1, 3))
#         #         ax.set_yticklabels(np.arange(1, x.shape[0] + 1, 1))
#         #         ax.grid(axis='y')
#
#         ax.set_title(title)
#         ax.set_xlim([0, x.shape[1] * dt])
#         ax.set_ylim([frm, tll])