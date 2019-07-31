import numpy as np
import pylab as plt

COLOR_ABBREVIATIONS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
PICKS_COLORMAP = plt.cm.tab10

TRACE_COLORS = {
    0: [.6, 0, 0],
    1: [0, .6, 0],
    2: [0, 0, .6],
    3: [0, 0, 0],
}
FILL_COLORS = {
    0: [.9, 0, 0],
    1: [0, .9, 0],
    2: [0, 0, .9],
    3: [.5, .5, .5],
}


def cast_input_to_traces(x, ndmin=2):
    """
    It is better to check input data type and shape. To prevent problems, array must be at least 2D,
    where time axis equals 1.
    :param x: list or array
    :param ndmin: minimal array dimension
    :return: np.array(x, ndmin=ndmin)
    """
    # x = np.array(x)
    # x = np.squeeze(x)
    x = np.array(x)
    x = x.copy()
    x = np.array(x, ndmin=ndmin, dtype=np.float32)
    return x


def cast_input_to_trace(x):
    """
    It is better to check input data type and shape. To prevent problems, array must be at least 1D,
    where time axis equals 1.
    :param x: list or array
    :param ndmin: minimal array dimension
    :return: np.array(x, ndmin=ndmin)
    """
    return cast_input_to_traces(x, ndmin=1)


def insert_zeros_in_trace(trace):
    trace = cast_input_to_trace(trace) + 1e-16
    # trace = np.squeeze(trace)

    time = np.arange(len(trace))
    zero_idx = np.where(np.diff(np.signbit(trace)))[0]

    time_at_zero = time[zero_idx] - trace[zero_idx] / np.diff(trace)[zero_idx]

    trace_z = np.insert(trace, zero_idx+1, 0)
    time_z = np.insert(time, zero_idx+1, time_at_zero)

    return trace_z, time_z


def input_check(traces, offset, mask, picks, ndmin=2):
    type_error = "'{}' must be a numpy array. the given was '{}'"
    shape_error = "'{}' must have same receivers as 'traces' receivers. the given was '{}'"

    if type(traces).__module__ != np.__name__:
        raise TypeError(type_error.format('traces', type(traces)))
    traces = cast_input_to_traces(traces, ndmin=ndmin)

    if len(traces.shape) != 3:
        traces = traces[..., None]

    if not isinstance(offset, type(None)):
        if isinstance(offset, list) | isinstance(offset, tuple):
            offset = np.array(offset)

        if type(offset).__module__ != np.__name__:
            raise TypeError(type_error.format('offset', type(offset)))

        # offset = cast_input_to_trace(offset)

        if offset.shape[0] != traces.shape[0]:
            raise ValueError(shape_error.format('offset', offset.shape))
    else:
        offset = np.arange(1, traces.shape[0] + 1)

    if not isinstance(mask, type(None)):
        if type(mask).__module__ != np.__name__:
            raise TypeError(type_error.format('mask', type(mask)))

        mask = cast_input_to_traces(mask, ndmin=ndmin)

        if mask.shape[0] != traces.shape[0]:
            raise ValueError(shape_error.format('mask', mask.shape))

        if mask.shape[1] != traces.shape[1]:
            raise Warning(
                "'mask' better have same sampling as 'traces' sampling. "
                "The given 'mask' was {}."
                "The given 'traces' was {}".format(
                    mask.shape, traces.shape
                )
            )

    if isinstance(picks, type(None)):
        picks = {}
    else:
        if not isinstance(picks, dict):
            picks = {
                'picks': picks
            }

    for label in picks:
        picks[label] = np.array(picks[label], dtype=np.int32)
        picks[label] = np.squeeze(picks[label])
        if picks[label].shape[0] != traces.shape[0]:
            raise ValueError(shape_error.format('picks - "{}""'.format(label), picks[label].shape))

    return traces, offset, mask, picks


def input_check_color_dicts(no_of_components, **kwargs):
    result = {}
    for dict_name, color_dict in kwargs.items():
        if dict_name == 'trace_color':
            if isinstance(color_dict, type(None)) | isinstance(color_dict, bool):
                result[dict_name] = TRACE_COLORS
            else:
                result[dict_name] = color_dict
        else:
            if isinstance(color_dict, type(None)):
                result[dict_name] = {}
                continue

            elif isinstance(color_dict, bool):
                if color_dict:
                    result[dict_name] = FILL_COLORS
                else:
                    result[dict_name] = {}
                    continue
            else:
                result[dict_name] = color_dict

        if not isinstance(result[dict_name], dict):
            result[dict_name] = {x: color_dict for x in range(no_of_components)}

        for i in range(no_of_components):
            if i not in result[dict_name]:
                raise ValueError('No color given for component "{}"'.format(i))

        for label, color in result[dict_name].items():
            if isinstance(color, str):
                if not (color in COLOR_ABBREVIATIONS):
                    raise ValueError(
                        "'{}' items must be one of {}"
                        " or an array like [r, g, b] or [r, g, b, a]. "
                        "The given was '{}'".format(dict_name, COLOR_ABBREVIATIONS, color)
                    )
            else:
                color = np.array(color).astype(float)
                if len(color) not in [3, 4]:
                    raise ValueError(
                        "'{}' RGBA sequence should have length 3 or 4. "
                        "The given was '{}".format(dict_name, color)
                    )

    return result.values()


def input_chek_picks_color(picks, picks_colormap, picks_line_style, picks_curve_line_style):

    if isinstance(picks_line_style, type(None)):
        picks_line_style = 'dashed'

    if isinstance(picks_curve_line_style, type(None)):
        picks_curve_line_style = 'solid'

    if isinstance(picks_colormap, type(None)):
        return {x: PICKS_COLORMAP(i) for i, x in enumerate(picks)}, picks_line_style, picks_curve_line_style

    if isinstance(picks_colormap, str):
        if not (picks_colormap in COLOR_ABBREVIATIONS):
            raise ValueError(
                "'{}' items must be one of {}"
                " or an array like [r, g, b] or [r, g, b, a]. "
                "The given was '{}'".format('picks_colormap', COLOR_ABBREVIATIONS, picks_colormap)
            )
        else:
            return {x: picks_colormap for x in picks}, picks_line_style, picks_curve_line_style

    if type(picks_colormap).__name__ == 'ListedColormap':
        return {x: picks_colormap(i) for i, x in enumerate(picks)}, picks_line_style, picks_curve_line_style

    raise ValueError('the given colormap is undefined - "{}"'.format(picks_colormap))


