import numpy as np


def _moving_average_left(signal, window):
    ns = len(signal)
    average = np.zeros(signal.shape)
    for i in range(window, ns):
        temp = signal[(i-window):i]
        average[i] = temp.mean()
    return average


def _moving_average_right(signal, window):
    ns = len(signal)
    average = np.zeros(signal.shape)
    for i in range(ns - window):
        temp = signal[i:(i+window)]
        average[i] = temp.mean()
    return average


WINDOW_TYPE_AVERAGING = {'left': _moving_average_left,
                         'right': _moving_average_right}


def moving_average(signal, window_type='right', window=1):
    ns = len(signal)
    if window <= 0:
        print('Длина окна должна быть больше нуля')
        return signal
    if window >= ns:
        print('Длина окна больше длины сигнала')
        return signal

    averaging_func = WINDOW_TYPE_AVERAGING[window_type]
    average = averaging_func(signal, window)

    return average
