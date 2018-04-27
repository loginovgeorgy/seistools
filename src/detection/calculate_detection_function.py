from .detection_functions import _DETECTION_FUNCTIONS


def calculate_function(signal,
                       function_type='mer',
                       **parameters):

    detection_function = _DETECTION_FUNCTIONS[function_type]
    d = detection_function(signal, **parameters)
    return d