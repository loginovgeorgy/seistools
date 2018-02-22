import numpy as np
from moving_average.moving_average import moving_average


def _detection_function_stalta(signal, ns, nl, eps=1e-16):
    sta = moving_average(signal ** 2, window=ns, window_type='left')
    lta = moving_average(signal ** 2, window=nl, window_type='left')
    # TODO: придумать метод определения eps (регуляризация)
    lta = lta + eps
    d = sta / lta
    d[0:nl] = 1
    d_diff = np.gradient(d)

    detection = np.abs(d_diff)

    return detection


def _detection_function_mer(signal, ns, eps=1e-16):
    mer_window1 = moving_average(signal ** 2, window=ns, window_type='left')
    mer_window2 = moving_average(signal ** 2, window=ns, window_type='right')
    mer_window2 = mer_window2 + eps
    d = mer_window1 / mer_window2
    
    return d


def _detection_function_em(signal, ns):
    em_window = moving_average(signal ** 2, window=ns, window_type='left')

    return em_window


_DETECTION_FUNCTIONS = {'stalta': _detection_function_stalta,
                        'mer': _detection_function_mer,
                        'em': _detection_function_em}

