from .seislet_functions import *

LET_FUNCTIONS = {'berlage': berlage_function,
                 'puzirev': puzirev_function,
                 'chirplet': chirplet_function,
                 'ricker': ricker_function}


def gen_seislet(seislet,
                dt=DEFAULT_DT,
                ns=DEFAULT_NS,
                **kwargs):
    signal_function = LET_FUNCTIONS[seislet]
    signal = signal_function(dt, ns, **kwargs)
    return signal 