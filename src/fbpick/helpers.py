import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.signal import convolve


def cast_input_to_array(x, ndmin=2):
    """
    It is better to check input data type and shape. To prevent problems, array must be at least 2D,
    where time axis equals 1.
    :param x: list or array
    :param ndmin: minimal array dimension
    :return: np.array(x, ndmin=ndmin)
    """
    # x = np.array(x)
    # x = np.squeeze(x)
    x = np.array(x, ndmin=ndmin, dtype=np.float32)
    return x


def add_zeros(data, nr, ns):
    data = deepcopy(data)
    data = cast_input_to_array(data)

    if data.shape[0] < nr:
        temp = np.zeros((nr - data.shape[0], data.shape[1]))
        data = np.vstack((data, temp))
    if data.shape[1] < ns:
        temp = np.zeros((data.shape[0], ns - data.shape[1]))
        data = np.hstack((data, temp))
    if data.shape[0] > nr:
        data = data[:nr]
    if data.shape[1] > ns:
        data = data[:ns]
    return data


def matrix_delta(tau, ns=None):
    """
    Create matrix of delta functions at samples tau
    :param tau: array of int
    :param ns: maximum number of samples, must be higher then 0
    :return:
        array of shape len(tau) * ns
    """
    tau = np.array(tau)
    tau = np.int32(tau)
    if ns:
        ns = np.int32(ns)
        if ns < 1:
            raise Exception(
                'ns must be more then 0.'
                'The given ns was {}'.format(ns)
            )

        tau[tau >= ns] = ns - 1
    else:
        ns = tau.max()

    ndmin = len(tau.shape)

    matrix = np.zeros(list(tau.shape) + [ns])
    slc_w = [slice(None)] * (ndmin + 1)

    for i in range(ndmin):
        tmp = np.arange(tau.shape[i])
        tmp = np.array(tmp, ndmin=ndmin)
        tmp = np.transpose(tmp, np.roll(np.arange(ndmin), -(i + ndmin - 1)))

        slc_w[i] = tmp

    slc_w[ndmin] = tau

    matrix[tuple(slc_w)] = 1
    return matrix


def matrix_heaviside(tau, ns=None):
    """
    Create matrix of heaviside functions at samples tau
    :param tau: array of int
    :param ns: maximum number of samples, must be higher then 0
    :return:
        array of shape len(tau) * ns
    """
    matrix = matrix_delta(tau, ns)
    return np.cumsum(matrix, axis=1)


def calculate_travel_time(offset, vel0=2000, dt=.002):
    return np.int32(np.round(offset / dt / vel0))


def moving_average_1d(x, window, axis=1, window_type='left'):
    """
    Calculate Moving Average of signal along certain axis.
    :param x: input array
    :param window: window length
    :param axis:
    :param window_type:
        'left'
        'right'
        'center'
    :return:
    """

    window = np.int32(window)

    if window_type not in ['left', 'right', 'center']:
        raise Exception(
            'Must choose one of three window types: "left", "right", "center"'
            'The given window type is {}'.format(window_type)
        )

    if (not window % 2) & (window_type == 'center'):
        raise Exception(
            'window length must be odd.'
            'The window was {}.'.format(window)
        )

    x = deepcopy(x)
    x = cast_input_to_array(x, ndmin=2)
    ns = x.shape[axis]

    f = np.ones(window)/window
    f = cast_input_to_array(f, ndmin=len(x.shape))
    f = f.transpose(
        np.roll(
            np.arange(len(x.shape)),
            axis + 1
        )
    )

    if window_type == 'center':
        mode = 'same'
    else:
        mode = 'full'

    if window_type == 'left':
        x = np.flip(x, axis=axis)

    average = convolve(x, f, mode=mode)

    slc = [slice(None)] * len(x.shape)

    if window_type == 'left':
        average = np.flip(average, axis=axis)

    if window_type == 'right':
        slc[axis] = slice(window - 1, ns + window)
    else:
        slc[axis] = slice(0, ns)

    return average[tuple(slc)]


def edge_preserve_smoothing(signal, window, verbose=False):
    if window<=0:
        return signal
    ns = len(signal)
    eps_avg = np.zeros(ns)
    for i in range(ns):
        jfrom = max([i - window + 1, 0])
        interval = np.arange(jfrom, i+1)
        _avg = np.zeros(len(interval))
        _std = np.zeros(len(interval))
        if verbose:
            print(i, interval)
        for j, jfrom in enumerate(interval):
            jto = min([jfrom + window, ns])

            temp_signal = signal[jfrom:jto]
            _avg[j] = temp_signal.mean()
            _std[j] = temp_signal.std()
            if verbose:
                print("\t {} {:.2f} {:.2f}".format(str((np.arange(jfrom,jto))), _avg[j], _std[j]))

#         eps_avg[i] = _avg[np.where(_std==_std.min())[0]].min()
        eps_avg[i] = _avg[_std.argmin()]
    return eps_avg


def binning_column(df, column='OFFSET', bins=50, merge=True, suffix='BIN', label=True):
    """
    Function provides binning (cut column by bins).
    :param df: Header or any other DataFrame
    :param column: col. to provide binning
    :param bins: list/array/None. use certain bins if passed by user
    :param step: int. use step for binning (np.arange(df[column].min(), df[column].max() + step, step))
    :param merge: True/False
    :param suffix: str of how to call new_column - new_column = "_".join([column, suffix])
    :param label: True/False. add bin labels
    :return:
        'pd.cut' if merge=False
        'pd.merge(df, pd.cut)' if merge=True
    """
    _bins = deepcopy(bins)
    df_tmp, bins = pd.cut(df[column], _bins, retbins=True)
    if not merge:
        return df_tmp

    new_column = "_".join([column, suffix])
    df[new_column] = df_tmp
    mid = [(a + b) / 2 for a, b in zip(bins[:-1], bins[1:])]

    if label:
        new_column = "_".join([column, suffix, 'LABEL'])
        df[new_column] = df_tmp.cat.rename_categories(mid).round(1)

    return df


def hodograph(rec, vel, thk, ang, n_bound):
    """
    Calculate  hodograph
    """
    t = rec / vel[n_bound]

    x_gol = 0
    i = 0
    p = 1

    while i < n_bound:

        if n_bound == 3:
            p = 0

        t = t + 2 * thk[i] * np.cos(ang[n_bound - p + i]) / vel[i]
        x_gol = x_gol + 2 * thk[i] * np.tan(ang[n_bound - p + i])
        i += 1

    xg = np.floor(x_gol / 10)

    print(xg)
    t[0:xg] = None

    return t


def calculate_statistics(df, group_by_col, column='FB', method='mean', merge=True):
    """
    Group and calculate statistics
    :param df:
    :param group_by_col: str or list
    :param column: str
    :param method: 'mean' | 'median' | 'std' | 'sum'
    :param merge: True/False
    :return:
    """
    if not isinstance(group_by_col, list):
        group_by_col = [group_by_col]

    df = df.copy()
    new_column = '_'.join(group_by_col + [column] + [method.upper()])

    df_grouped = df.groupby(group_by_col)
    df_grouped = df_grouped.agg({column: method})
    df_grouped = df_grouped.rename(columns={column: new_column})
    #     df_grouped = df_grouped.to_frame(new_col)
    df_grouped = df_grouped.fillna(0)

    if not merge:
        return df_grouped

    df = df[np.setdiff1d(df.columns, [new_column])]
    df = pd.merge(df, df_grouped, left_on=group_by_col, right_index=True)
    return df


def merge_n_replace_left_by_right(df_left, df_right, on='IDX'):

    if on not in df_left.columns:
        raise Exception('Column "{}" not in df_left'.format(on))

    if on not in df_right.columns:
        raise Exception('Column "{}" not in df_right'.format(on))

    df_left = df_left.copy()
    df_right = df_right.copy()
    left_columns = np.setdiff1d(df_left.columns, df_right.columns).tolist() + [on]
    return pd.merge(df_left[left_columns], df_right, on=on)
